# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import sys, os
import numpy as np
import argparse
import json
from tqdm import tqdm

def get_transform_files(data_dir):
    transform_files = []
    for root, dirs, files in os.walk(data_dir):
        transform_files += [os.path.join(root, file) for file in files if file.endswith('transforms.json') or file.endswith('transforms_ego.json')]
    return transform_files

def split_dataset(data_dir, split_ratio):
    transform_files = get_transform_files(data_dir)
    
    num_files = len(transform_files)
    pbar = tqdm(transform_files, total=num_files)

    for file in pbar:
        with open(file, 'r') as f:
            transforms = json.load(f)
        
        train = transforms.copy()
        test = transforms.copy()
        frames = transforms["frames"]

        num_frames = len(frames)

        num_train_frames = int(num_frames * split_ratio)

        np.random.shuffle(frames)

        train["frames"] = frames[:num_train_frames]
        test["frames"] = frames[num_train_frames:]

        train_file = file.replace('.json', '_train.json')
        test_file = file.replace('.json', '_test.json')

        with open(train_file, 'w') as f:
            json.dump(train, f, indent=4)

        with open(test_file, 'w') as f:
            json.dump(test, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)

    split_dataset(args.data_dir, args.split_ratio)