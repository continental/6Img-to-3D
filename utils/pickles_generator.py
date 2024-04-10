# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import os

import numpy as np
import torch
from tqdm import tqdm
import argparse
from mmcv import Config

from multiprocessing import Pool

import triplane_decoder.rays_dataset as rays_dataset



class Triplane_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config):
        self.data_path = dataset_config.data_path
        self.dataset_config = dataset_config.train_data_loader
        self.data = []


        if dataset_config.get("town", "all") == "all":
            towns = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]
        else:
            towns = dataset_config["town"]
        for town in towns:
            if dataset_config.get("weather", "all")  == "all" :
                weathers = [folder for folder in os.listdir(os.path.join(self.data_path, town)) if os.path.isdir(os.path.join(self.data_path, town, folder))]
            else:
                weathers = dataset_config["weather"]
            for weather in weathers:
                if dataset_config.get("vehicle", "all")  == "all" :
                    vehicles = [folder for folder in os.listdir(os.path.join(self.data_path, town, weather)) if os.path.isdir(os.path.join(self.data_path, town, weather, folder))]
                else:
                    vehicles = dataset_config["vehicle"]
                for vehicle in vehicles:
                    if dataset_config.get("spawn_point", "all") == "all" :
                        spawn_points = [folder for folder in os.listdir(os.path.join(self.data_path, town, weather, vehicle)) if "spawn_point_" in folder]
                    else:
                        spawn_points = [f"spawn_point_{i}" for i in dataset_config["spawn_point"]]
                    for spawn_point in spawn_points:
                        if dataset_config.get("step", "all") == "all" :
                            steps = [folder for folder in os.listdir(os.path.join(self.data_path, town, weather, vehicle, spawn_point)) if "step_" in folder]
                        else:
                            steps = [f"step_{i}" for i in dataset_config["step"]]
                        for step in steps:
                            self.data.append(self.get_data(town, weather, vehicle, spawn_point, step))

    def get_data(self, town, weather, vehicle, spawn_point, step):
        data_path = os.path.join(self.data_path, town, weather, vehicle, spawn_point, step, "sphere")

        return data_path

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        
        return data

def process_datapath(datapath):
    try:
        train_dataset = rays_dataset.RaysDataset(datapath, config, dataset_config=dataset_config.train_data_loader, mode="train", factor=dataset_config.train_data_loader.factor)
        if dataset_config.train_data_loader.whole_image:
            W, H = train_dataset.intrinsics.width, train_dataset.intrinsics.height
            train_dataset = train_dataset.dataset.view(80, H*W, -1)
            for i in range(len(train_dataset)):
                with open(os.path.join(datapath, f"train_dataset_{i}.npy"), "wb") as f:
                    np.save(f, train_dataset[i].numpy())
    except Exception as e:
        return (False, datapath)
    return (True, datapath)



if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset-config", type=str, default="config/_base_/dataset.py")
    parser.add_argument("--py-config", type=str, default="config/config.py")


    args = parser.parse_args()

    dataset_config = Config.fromfile(args.dataset_config).dataset_params
    config = Config.fromfile(args.py_config)

    dataset_all = Triplane_Dataset(dataset_config)

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    failed = []

    
    with Pool(24) as p:
        results = list(tqdm(p.imap(process_datapath, dataset_all), total=len(dataset_all)))
        for success, datapath in results:
            if not success:
                failed.append(datapath)

    with open("failed.txt", "w") as f:
        for fail in failed:
            f.write(f"{fail}\n")