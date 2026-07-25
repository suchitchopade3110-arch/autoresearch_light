import os
import uuid
from typing import Dict, Tuple

import git


class MergeConflict(Exception):
    """
    Raised when a candidate branch can't be rebased cleanly onto the
    target branch. The rebase is aborted before this is raised, so the
    candidate's worktree is left clean (not mid-rebase) and its branch
    untouched - the caller is expected to roll it back, same as any other
    failure outcome.
    """
    def __init__(self, branch_name: str, message: str):
        super().__init__(f"Candidate branch {branch_name} could not be rebased cleanly: {message}")
        self.branch_name = branch_name
        self.message = message


class GitController:
    """
    Manages candidate branches using git worktrees, so a candidate's checkout
    always lives in its own directory. create_branch/commit_patch/rollback/merge
    never touch the caller's main working tree (or any uncommitted work in it),
    and each worktree is independent enough that concurrent candidates (see
    evolution/scheduler.py) don't need to share a checkout at all.
    """
    def __init__(self, repo_path: str = ".", worktree_root: str = None, base_ref: str = None):
        self.repo = git.Repo(repo_path)
        self.repo_path = os.path.abspath(repo_path)

        if base_ref:
            self.original_branch = base_ref
            self._tracks_named_branch = base_ref in [h.name for h in self.repo.heads]
        else:
            try:
                self.original_branch = self.repo.active_branch.name
                self._tracks_named_branch = True
            except TypeError:
                # Detached HEAD (e.g. a CI checkout of a specific commit):
                # there is no branch name to read or to rebase/merge onto,
                # so pin to the current commit instead of crashing here.
                self.original_branch = self.repo.head.commit.hexsha
                self._tracks_named_branch = False

        self.worktree_root = worktree_root or os.path.join(self.repo_path, ".candidate_worktrees")
        os.makedirs(self.worktree_root, exist_ok=True)

    def create_branch(self, candidate_id: str) -> Tuple[str, str]:
        """Creates a unique branch for the candidate in its own worktree. Returns (branch_name, worktree_path)."""
        branch_name = f"candidate-{candidate_id}-{uuid.uuid4().hex[:8]}"
        worktree_path = os.path.join(self.worktree_root, branch_name)
        self.repo.git.worktree("add", "-b", branch_name, worktree_path, self.original_branch)
        return branch_name, worktree_path

    def commit_patch(self, worktree_path: str, message: str = "Apply candidate patch") -> bool:
        """Commits all current changes within the candidate's worktree."""
        wt_repo = git.Repo(worktree_path)
        try:
            if not wt_repo.is_dirty(untracked_files=True):
                return False

            wt_repo.git.add(A=True)
            wt_repo.index.commit(message)
            return True
        finally:
            # Windows keeps a file lock on the worktree while this handle is
            # open, so a later `git worktree remove --force` on this same
            # path (in rollback/merge) fails with "Permission denied" unless
            # this is explicitly released first - Linux/macOS never enforce
            # that, which is why this only shows up on Windows.
            wt_repo.close()

    def rollback(self, branch_name: str, worktree_path: str) -> None:
        """Discards the candidate's worktree and deletes its branch. Never touches the main working tree."""
        if os.path.exists(worktree_path):
            self.repo.git.worktree("remove", "--force", worktree_path)
        self.repo.delete_head(branch_name, force=True)

    def cleanup_orphans(self) -> Dict[str, int]:
        """
        Removes every candidate worktree and branch left behind by a
        previous run that crashed (or was killed) before it could roll
        back or merge them. Safe to call unconditionally at startup: a
        freshly-started process never has any candidates of its own yet,
        so anything matching "candidate-*" under worktree_root, or a
        "candidate-*" branch, is necessarily orphaned leftovers.

        Never touches original_branch or the main worktree.
        """
        removed_worktrees = 0

        try:
            listing = self.repo.git.worktree("list", "--porcelain")
        except git.GitCommandError:
            listing = ""

        # A worktree is identified as an orphan candidate by the branch IT
        # HAS CHECKED OUT (parsed from git's own porcelain output), not by
        # string-comparing its path against worktree_root - Windows can
        # report the same directory in short (8.3, e.g. "SUCHIT~1") or long
        # form inconsistently between git and Python's os.path, which
        # silently breaks path-based comparison even though both refer to
        # the same directory on disk.
        orphan_paths = []
        current_path = None
        current_branch = None
        for line in listing.splitlines() + [""]:
            if line.startswith("worktree "):
                current_path = line[len("worktree "):]
            elif line.startswith("branch refs/heads/"):
                current_branch = line[len("branch refs/heads/"):]
            elif line == "":
                if (
                    current_path
                    and current_branch
                    and current_branch.startswith("candidate-")
                    and os.path.abspath(current_path) != self.repo_path
                ):
                    orphan_paths.append(current_path)
                current_path = None
                current_branch = None

        for path in orphan_paths:
            self.repo.git.worktree("remove", "--force", path)
            removed_worktrees += 1

        # Drops stale administrative files for worktrees whose directory was
        # already deleted from disk (e.g. a crash mid-removal), which the
        # loop above can't see since `worktree list` no longer reports them
        # as removable paths.
        self.repo.git.worktree("prune")

        removed_branches = 0
        for head in list(self.repo.heads):
            if head.name.startswith("candidate-") and head.name != self.original_branch:
                self.repo.delete_head(head.name, force=True)
                removed_branches += 1

        return {"removed_worktrees": removed_worktrees, "removed_branches": removed_branches}

    def merge(self, branch_name: str, worktree_path: str) -> None:
        """
        Rebases the candidate branch onto the target branch inside its own
        worktree first, then fast-forwards the target branch onto it - a
        fast-forward can never conflict after a clean rebase, so the
        shared main checkout can never be left in a half-merged state
        (no MERGE_HEAD, no conflict markers, repo.is_dirty() stays False).

        Raises MergeConflict if the rebase itself conflicts, leaving the
        candidate's worktree and branch in place for the caller to roll
        back - exactly like any other failure outcome. This is routine
        traffic in evolutionary mode, where several candidates rebase onto
        the same base per generation, not an edge case.
        """
        wt = git.Repo(worktree_path)
        try:
            wt.git.rebase(self.original_branch)
        except git.GitCommandError as e:
            try:
                wt.git.rebase("--abort")
            except git.GitCommandError:
                pass
            raise MergeConflict(branch_name, str(e))
        finally:
            wt.close()

        # git.checkout (not repo.heads[...].checkout()) works whether
        # original_branch is a branch name or a pinned commit sha
        # (detached HEAD case).
        self.repo.git.checkout(self.original_branch)
        self.repo.git.merge(branch_name, "--ff-only")

        if not self._tracks_named_branch:
            # A detached HEAD has no branch to auto-advance to the new tip.
            # Without this, the next candidate would rebase onto this
            # now-stale commit, and its own ff-only merge could silently
            # discard this one instead of building on it.
            self.original_branch = self.repo.head.commit.hexsha

        self.repo.git.worktree("remove", "--force", worktree_path)
        self.repo.delete_head(branch_name, force=True)
