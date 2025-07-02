from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import entropy
from tqdm import tqdm

PATH_LIST = [
    Path("saved/ens-1/prediction"),
    Path("saved/ens-2/prediction"),
    Path("saved/ens-3/prediction"),
]

SAVE_PATH = Path("saved/ensemble/prediction")

scans = sorted(list(PATH_LIST[0].glob("*/ens_*.npy")))


def process_scan(path, PATH_LIST, SAVE_PATH):
    """Processes a single scan file and saves the entropy values."""
    sequence = path.parent.stem
    file = path.name

    confid_list = []
    for model in PATH_LIST:
        filepath = model / sequence / file
        confid_list.append(np.load(filepath))

    all_confids = np.stack(confid_list)
    avg_probs = np.mean(all_confids, 0)
    entropy_values = entropy(avg_probs, axis=1) / np.log(avg_probs.shape[1])

    if not (SAVE_PATH / sequence).exists():
        (SAVE_PATH / sequence).mkdir(parents=True, exist_ok=True)
    np.save(SAVE_PATH / sequence / file, entropy_values)
    return None  # Return None to avoid collecting results


if __name__ == "__main__":
    Parallel(n_jobs=8)(
        delayed(process_scan)(path, PATH_LIST, SAVE_PATH) for path in tqdm(scans)
    )
