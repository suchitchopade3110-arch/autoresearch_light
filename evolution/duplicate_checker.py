from memory.db import ExperimentDB

def is_duplicate(diff: str, db: ExperimentDB, threshold: float = 0.25, hypothesis: str = "") -> bool:
    if not diff or diff.strip() == "":
        return False

    # Cheap exact-match check first - has_exact_diff is a metadata filter
    # (no embedding function invoked), unlike the nearest-neighbor search
    # below. Skips paying for an embedding + similarity search just to
    # then find out the closest result is a byte-for-byte repeat. A diff
    # that only differs from a stored one by leading/trailing whitespace
    # still falls through to the slower path below, which already handles
    # that case via a stripped string comparison.
    if db.has_exact_diff(diff):
        return True

    # Shape the query the same way store_experiment shapes its embedded
    # document (Hypothesis + Diff) - querying with the bare diff against a
    # document that also contains hypothesis/rationale/outcome text dilutes
    # the comparison so much that even an exact-diff match scores no better
    # than an unrelated one. Matching the shape (hypothesis included) brings
    # near-duplicate diffs to a distinctly lower distance than unrelated ones.
    query = f"Hypothesis: {hypothesis}\nDiff:\n{diff}" if hypothesis else diff

    results = db.retrieve_experiments(query=query, k=1)

    if not results:
        return False

    closest = results[0]
    distance = closest.get('distance', float('inf'))

    if diff.strip() == closest['diff'].strip():
        return True

    if distance < threshold:
        return True

    return False
