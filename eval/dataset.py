import json
import os
import random
from typing import Dict, List

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 1000
DEFAULT_TEST_FRAC = 0.25


def _generate_rows(n_samples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        x1 = rng.uniform(-1, 1)
        x2 = rng.uniform(-1, 1)
        noise = rng.uniform(-0.3, 0.3)
        label = 1 if (2 * x1 - x2 + noise) > 0 else 0
        rows.append({"x1": x1, "x2": x2, "label": label})
    return rows


def generate_split(dir_path: str, n: int = DEFAULT_N_SAMPLES, seed: int = DEFAULT_SEED,
                    test_frac: float = DEFAULT_TEST_FRAC) -> Dict[str, str]:
    """
    Writes a deterministic synthetic binary-classification dataset to
    `dir_path`, split so the ground truth for the held-out set never
    reaches the sandbox:

      - train.jsonl: {"x1","x2","label"} - mounted read-only into the sandbox
      - test.jsonl:  {"id","x1","x2"} (no label) - mounted read-only
      - truth.json:  {id: label} - stays on the host, never mounted

    A candidate that only ever sees train.jsonl and test.jsonl cannot read
    the held-out labels off disk, so a prediction score against truth.json
    reflects real generalization rather than a printed claim. Skips
    regeneration if train.jsonl already exists, same as the old
    generate_dataset's no-clobber contract.
    """
    train_path = os.path.join(dir_path, "train.jsonl")
    test_path = os.path.join(dir_path, "test.jsonl")
    truth_path = os.path.join(dir_path, "truth.json")

    if os.path.exists(train_path):
        return {"train": train_path, "test": test_path, "truth": truth_path}

    os.makedirs(dir_path, exist_ok=True)

    rows = _generate_rows(n, seed)
    n_test = max(1, int(len(rows) * test_frac))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    test_idx = set(order[:n_test])

    truth = {}
    with open(train_path, "w") as train_f, open(test_path, "w") as test_f:
        for i, row in enumerate(rows):
            if i in test_idx:
                test_f.write(json.dumps({"id": i, "x1": row["x1"], "x2": row["x2"]}) + "\n")
                truth[str(i)] = row["label"]
            else:
                train_f.write(json.dumps(row) + "\n")

    with open(truth_path, "w") as f:
        json.dump(truth, f)

    return {"train": train_path, "test": test_path, "truth": truth_path}


def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_truth(path: str) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


def subset_indices(n: int, percentage: float, seed: int = DEFAULT_SEED) -> List[int]:
    """
    Deterministic index selection such that subsets at smaller percentages
    are always contained within subsets at larger percentages - so
    progressive-scaling stages see nested, not independently-sampled, data.
    Only ever applied to train.jsonl - the test set stays whole at every
    stage so scores stay comparable across stages.
    """
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    k = max(1, int(n * percentage / 100))
    return order[:k]


def load_subset(path: str, percentage: float, seed: int = DEFAULT_SEED) -> List[Dict]:
    rows = load_dataset(path)
    idx = subset_indices(len(rows), percentage, seed)
    return [rows[i] for i in idx]
