"""
This plays the role of loading Spriv in the paper: Spriv = {(x_i, y_i)}.
x_i is a path to a .wav file and y_i is one of the 10 class names.
"""

__version__ = "v3"

import csv
import os
import random
from collections import defaultdict


def load_urbansound8k(root: str) -> dict:
    """
    returns dictionary where each key is an audio class name and each value is list of all audio file paths in that class
    """
    csv_path = os.path.join(root, "metadata", "UrbanSound8K.csv")
    class_to_paths = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold = row["fold"]                 # which of the 10 predefined folds this clip is in
            fname = row["slice_file_name"]     # the .wav filename
            class_name = row["class"]          # e.g. "dog_bark"
            wav_path = os.path.join(root, "audio", f"fold{fold}", fname)
            class_to_paths[class_name].append(wav_path)

    return dict(class_to_paths)


def train_test_split_by_fold(class_to_paths: dict, test_folds=(10,)) -> tuple:
    train, test = defaultdict(list), defaultdict(list)  # train = private dataset, test = held-out real data for downstream-classifier accuracy check
    for class_name, paths in class_to_paths.items():
        for p in paths:
            fold_num = int(p.split("/fold")[1].split("/")[0])   # extract fold number out of path string -> if fold number is in test_folds, put path in test; if not, put in train
            (test if fold_num in test_folds else train)[class_name].append(p)
    return dict(train), dict(test)


def subsample_per_class(class_to_paths: dict, max_per_class: int, seed: int = 0) -> dict:
    """
    randomly keeps at most max_per_class paths per class
    """
    if max_per_class is None:
        return class_to_paths

    rng = random.Random(seed)   # local Random instance so this doesn't disturb global random state used elsewhere
    subsampled = {}
    for class_name, paths in class_to_paths.items():
        if len(paths) <= max_per_class:
            subsampled[class_name] = paths
        else:
            subsampled[class_name] = rng.sample(paths, max_per_class)
    return subsampled
