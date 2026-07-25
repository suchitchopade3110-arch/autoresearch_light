import os
import tempfile

import git
import pytest

from vcs.git_controller import GitController, MergeConflict


@pytest.fixture
def repo_dir():
    with tempfile.TemporaryDirectory() as d:
        repo = git.Repo.init(d)
        try:
            test_file = os.path.join(d, "test.txt")
            with open(test_file, "w") as f:
                f.write("initial state")
            repo.index.add(["test.txt"])
            repo.index.commit("initial commit")
        finally:
            # Windows keeps the tempdir locked while this handle is open,
            # which would fail the TemporaryDirectory cleanup below.
            repo.close()

        yield d


def test_create_branch_does_not_touch_main_worktree(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    branch_name, worktree_path = controller.create_branch("123")

    # main working tree stays checked out on the original branch throughout
    assert controller.repo.active_branch.name == original_branch
    assert os.path.isdir(worktree_path)

    wt_repo = git.Repo(worktree_path)
    try:
        assert wt_repo.active_branch.name == branch_name
    finally:
        wt_repo.close()


def test_git_controller_rollback(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    branch_name, worktree_path = controller.create_branch("123")
    wt_file = os.path.join(worktree_path, "test.txt")
    with open(wt_file, "w") as f:
        f.write("changed state")
    controller.commit_patch(worktree_path)

    # the candidate's change lives only in its own worktree
    with open(os.path.join(repo_dir, "test.txt")) as f:
        assert f.read() == "initial state"

    controller.rollback(branch_name, worktree_path)

    assert controller.repo.active_branch.name == original_branch
    assert branch_name not in [h.name for h in controller.repo.heads]
    assert not os.path.exists(worktree_path)


def test_git_controller_merge(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    branch_name, worktree_path = controller.create_branch("456")
    wt_file = os.path.join(worktree_path, "test.txt")
    with open(wt_file, "w") as f:
        f.write("changed state")
    controller.commit_patch(worktree_path)

    controller.merge(branch_name, worktree_path)

    assert controller.repo.active_branch.name == original_branch
    with open(os.path.join(repo_dir, "test.txt")) as f:
        assert f.read() == "changed state"
    assert branch_name not in [h.name for h in controller.repo.heads]
    assert not os.path.exists(worktree_path)


def test_rollback_never_touches_uncommitted_changes_in_main_worktree(repo_dir):
    """
    Regression test: the pre-worktree implementation checked candidate
    branches out directly in the caller's repo and hard-reset + git-clean'd
    it on rollback, which would destroy any uncommitted work sitting in the
    caller's working tree. Worktree isolation must prevent that.
    """
    controller = GitController(repo_dir)

    scratch_file = os.path.join(repo_dir, "scratch.txt")
    with open(scratch_file, "w") as f:
        f.write("uncommitted work in progress")

    branch_name, worktree_path = controller.create_branch("789")
    with open(os.path.join(worktree_path, "test.txt"), "w") as f:
        f.write("candidate change")
    controller.commit_patch(worktree_path)

    controller.rollback(branch_name, worktree_path)

    assert os.path.exists(scratch_file)
    with open(scratch_file) as f:
        assert f.read() == "uncommitted work in progress"


def test_second_conflicting_merge_raises_and_leaves_the_main_checkout_clean(repo_dir):
    """
    Wave 3 acceptance: two candidates that touch the same lines both get
    merged - the second must raise MergeConflict (its rebase can't apply
    cleanly onto what the first one already merged), and the main checkout
    must stay completely clean throughout: no MERGE_HEAD, not dirty.
    """
    # worktree_root outside repo_dir - a worktree_root left under the
    # default (inside the repo) would show up as untracked content in
    # is_dirty() below, which is a test-repo artifact (the real repo
    # gitignores .candidate_worktrees/), not something this test is about.
    with tempfile.TemporaryDirectory() as worktree_root:
        controller = GitController(repo_dir, worktree_root=worktree_root)

        branch_a, worktree_a = controller.create_branch("aaa")
        with open(os.path.join(worktree_a, "test.txt"), "w") as f:
            f.write("changed by A")
        controller.commit_patch(worktree_a)

        branch_b, worktree_b = controller.create_branch("bbb")
        with open(os.path.join(worktree_b, "test.txt"), "w") as f:
            f.write("changed by B")
        controller.commit_patch(worktree_b)

        # First candidate merges cleanly - fast-forward, no conflict.
        controller.merge(branch_a, worktree_a)
        with open(os.path.join(repo_dir, "test.txt")) as f:
            assert f.read() == "changed by A"

        # Second candidate's rebase onto the new base conflicts (same line,
        # already changed) - must raise, not silently corrupt the checkout.
        with pytest.raises(MergeConflict) as exc_info:
            controller.merge(branch_b, worktree_b)
        assert branch_b in str(exc_info.value)

        # Main checkout is untouched: no merge in progress, nothing dirty.
        assert controller.repo.is_dirty(untracked_files=True) is False
        assert not os.path.exists(os.path.join(controller.repo.git_dir, "MERGE_HEAD"))
        with open(os.path.join(repo_dir, "test.txt")) as f:
            assert f.read() == "changed by A"

        # The failed candidate's branch/worktree are left for the caller to
        # roll back, exactly like any other failure outcome.
        assert branch_b in [h.name for h in controller.repo.heads]
        controller.rollback(branch_b, worktree_b)
        assert branch_b not in [h.name for h in controller.repo.heads]
        assert not os.path.exists(worktree_b)


def test_detached_head_does_not_crash_init_and_can_still_create_branches(repo_dir):
    """
    Wave 3 acceptance: a target repo checked out at a specific commit
    (detached HEAD, e.g. a CI checkout) must not crash GitController.__init__
    via repo.active_branch.name (which raises TypeError when detached) -
    it should fall back to the current commit sha.
    """
    repo = git.Repo(repo_dir)
    try:
        repo.git.checkout(repo.head.commit.hexsha)  # detach HEAD
        assert repo.head.is_detached
    finally:
        repo.close()

    controller = GitController(repo_dir)
    assert controller.original_branch == controller.repo.head.commit.hexsha

    branch_name, worktree_path = controller.create_branch("det")
    assert os.path.isdir(worktree_path)
    with open(os.path.join(worktree_path, "test.txt")) as f:
        assert f.read() == "initial state"


def test_detached_head_merge_advances_pinned_commit_without_losing_prior_merges(repo_dir):
    """
    After a successful merge while detached, there is no branch to
    auto-advance - original_branch must be refreshed to the new tip so a
    second merge builds on the first, instead of silently discarding it by
    rebasing/ff-only-merging back onto the stale pre-merge commit.
    """
    repo = git.Repo(repo_dir)
    try:
        repo.git.checkout(repo.head.commit.hexsha)
    finally:
        repo.close()

    controller = GitController(repo_dir)

    branch_a, worktree_a = controller.create_branch("aaa")
    with open(os.path.join(worktree_a, "a.txt"), "w") as f:
        f.write("from a")
    controller.commit_patch(worktree_a)
    controller.merge(branch_a, worktree_a)

    branch_b, worktree_b = controller.create_branch("bbb")
    with open(os.path.join(worktree_b, "b.txt"), "w") as f:
        f.write("from b")
    controller.commit_patch(worktree_b)
    controller.merge(branch_b, worktree_b)

    assert os.path.exists(os.path.join(repo_dir, "a.txt"))
    assert os.path.exists(os.path.join(repo_dir, "b.txt"))


def test_base_ref_pins_a_specific_commit_regardless_of_current_branch(repo_dir):
    """target.base_ref must let candidates be based on an explicit ref even
    while the repo's checked-out branch is something else."""
    repo = git.Repo(repo_dir)
    try:
        pinned_sha = repo.head.commit.hexsha
        repo.git.checkout("-b", "other-branch")
        with open(os.path.join(repo_dir, "test.txt"), "w") as f:
            f.write("advanced on other-branch")
        repo.index.add(["test.txt"])
        repo.index.commit("advance other-branch")
    finally:
        repo.close()

    controller = GitController(repo_dir, base_ref=pinned_sha)
    assert controller.original_branch == pinned_sha

    branch_name, worktree_path = controller.create_branch("pinned")
    with open(os.path.join(worktree_path, "test.txt")) as f:
        assert f.read() == "initial state"


def test_cleanup_orphans_removes_leftover_worktrees_and_branches_from_a_crashed_run(repo_dir):
    """
    Wave 3 acceptance: crash recovery. A candidate created by a prior run
    that crashed before rollback/merge ran leaves its worktree and branch
    behind - a fresh GitController (simulating the next process start)
    must be able to prune them without disturbing the original branch or
    any worktree outside its own worktree_root.
    """
    with tempfile.TemporaryDirectory() as worktree_root:
        crashed_controller = GitController(repo_dir, worktree_root=worktree_root)
        branch_name, worktree_path = crashed_controller.create_branch("orphan")
        with open(os.path.join(worktree_path, "test.txt"), "w") as f:
            f.write("in-flight when the process died")
        crashed_controller.commit_patch(worktree_path)
        # Simulate a crash: no rollback(), no merge() - branch/worktree
        # just sit there, exactly as they would after a killed process.

        fresh_controller = GitController(repo_dir, worktree_root=worktree_root)
        original_branch = fresh_controller.original_branch

        result = fresh_controller.cleanup_orphans()

        assert result == {"removed_worktrees": 1, "removed_branches": 1}
        assert not os.path.exists(worktree_path)
        assert branch_name not in [h.name for h in fresh_controller.repo.heads]
        assert original_branch in [h.name for h in fresh_controller.repo.heads]
        assert fresh_controller.repo.active_branch.name == original_branch
        with open(os.path.join(repo_dir, "test.txt")) as f:
            assert f.read() == "initial state"


def test_cleanup_orphans_is_a_noop_when_nothing_is_orphaned(repo_dir):
    controller = GitController(repo_dir)
    result = controller.cleanup_orphans()
    assert result == {"removed_worktrees": 0, "removed_branches": 0}
    assert controller.repo.active_branch.name == controller.original_branch


def test_two_concurrent_candidates_get_independent_worktrees(repo_dir):
    """The evolutionary scheduler relies on this: candidates never share a checkout."""
    controller = GitController(repo_dir)

    branch_a, worktree_a = controller.create_branch("aaa")
    branch_b, worktree_b = controller.create_branch("bbb")

    assert worktree_a != worktree_b
    # each candidate edits its own new file, so both merges apply cleanly -
    # this test is about checkout isolation, not merge-conflict resolution
    with open(os.path.join(worktree_a, "a.txt"), "w") as f:
        f.write("from a")
    with open(os.path.join(worktree_b, "b.txt"), "w") as f:
        f.write("from b")

    assert not os.path.exists(os.path.join(worktree_a, "b.txt"))
    assert not os.path.exists(os.path.join(worktree_b, "a.txt"))

    controller.commit_patch(worktree_a)
    controller.commit_patch(worktree_b)
    controller.merge(branch_a, worktree_a)
    controller.merge(branch_b, worktree_b)

    assert os.path.exists(os.path.join(repo_dir, "a.txt"))
    assert os.path.exists(os.path.join(repo_dir, "b.txt"))
