# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import os
import numpy as np
from torch.utils import data
from mmcv.image.io import imread
import json
import random
from dataloader.transform_3d import NormalizeMultiviewImage
from torch.utils.data import DataLoader
from dataloader.rays_dataset import RaysDataset

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class CarlaDataset(data.Dataset):
    def __init__(self, data_path, dataset_config, config):
        self.data_path = data_path
        self.dataset_config = dataset_config
        self.data = []
        self.config = config
        self.decoder_config = config.decoder
        self.hw = (48,64)

        self.transforms = NormalizeMultiviewImage(**img_norm_cfg)
                            

        if dataset_config["town"] == "all":
            towns = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
        else:
            towns = dataset_config["town"]
        for town in towns:
            if dataset_config["weather"] == "all":
                weathers = [folder for folder in os.listdir(os.path.join(data_path, town)) if os.path.isdir(os.path.join(data_path, town, folder))]
            else:
                weathers = dataset_config["weather"]
            for weather in weathers:
                if dataset_config["vehicle"] == "all":
                    vehicles = [folder for folder in os.listdir(os.path.join(data_path, town, weather)) if os.path.isdir(os.path.join(data_path, town, weather, folder))]
                else:
                    vehicles = dataset_config["vehicle"]
                for vehicle in vehicles:
                    if dataset_config["spawn_point"] == ["all"]:
                        spawn_points = [folder for folder in os.listdir(os.path.join(data_path, town, weather, vehicle)) if "spawn_point_" in folder]
                    else:
                        spawn_points = [f"spawn_point_{i}" for i in dataset_config["spawn_point"]]
                    for spawn_point in spawn_points:
                        if dataset_config["step"] == ["all"]:
                            steps = [folder for folder in os.listdir(os.path.join(data_path, town, weather, vehicle, spawn_point)) if "step_" in folder]
                            steps = sorted(steps, key=lambda x: int(x.split('_')[1]))

                        else:
                            steps = [f"step_{i}" for i in dataset_config["step"]]
                        for step in steps:
                            self.data.append(self.get_data(town, weather, vehicle, spawn_point, step))

    def get_data(self, town, weather, vehicle, spawn_point, step):
        data_path = os.path.join(self.data_path, town, weather, vehicle, spawn_point, step)
        data = dict(
            town = town,
            weather = weather,
            vehicle = vehicle,
            spawn_point = spawn_point,
            step = step,
            nuscenes = os.path.join(data_path, "nuscenes"),
            sphere = os.path.join(data_path, "sphere"),
        )

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        
        imgs_dict = {
            "img": [],
        }

        img_meta = None
        input_rgb=np.empty((6,3,100,100))
        sphere_dataloader = None

        if "input_images" in self.dataset_config.get("selection", ["input_images"]):
            with open(os.path.join(data["nuscenes"], "transforms", "transforms_ego.json"), "r") as f:
                input_data = json.load(f)

            input_rgb = []
            all_c2w = []
            K = np.zeros((3,4)) # (C,3,4)
            K[0,0] = input_data['fl_x']
            K[1,1] = input_data['fl_y']
            K[2,2] = 1
            K[0,2] = input_data['cx']
            K[1,2] = input_data['cy']

            for frame in input_data["frames"]:
                input_rgb.append(imread(os.path.join(data["nuscenes"], "transforms", frame["file_path"]), "unchanged")[:,:,:3].astype(np.float32))
                all_c2w.append(frame["transform_matrix"])


            input_rgb = self.transforms(input_rgb)

            img_shape = [img.shape for img in input_rgb]

            img_meta = dict(
                K=K,
                c2w=all_c2w,
                img_shape=img_shape,
            )


        mode_suffix = "test" if self.dataset_config["phase"] == "val" else  self.dataset_config["phase"] 
        

        if "sphere_dataset" in self.dataset_config.get("selection", ["sphere_dataset"]):
            sphere_dataset = RaysDataset(data["sphere"], config=self.config, dataset_config=self.dataset_config, mode=mode_suffix, factor=self.dataset_config.factor)

            if self.dataset_config["phase"] == "train":
                sphere_dataloader = DataLoader(sphere_dataset, batch_size=self.dataset_config.get("batch_size",1), shuffle=True, num_workers=12, pin_memory=True)
            else:
                sphere_dataloader = DataLoader(sphere_dataset, batch_size=self.dataset_config.get("batch_size",1), shuffle=False)

        if "path" in self.dataset_config.get("selection", ["path"]):
            path = f"{data['town']}_{data['weather']}_{data['vehicle']}_{data['spawn_point']}_{data['step']}"
        else:
            path = None
       
        
        return (input_rgb, img_meta, sphere_dataloader) if path is None else (imgs_dict["img"], img_meta, sphere_dataloader, path)
    

class PickledCarlaDataset(CarlaDataset):

    def __init__(self, data_path, dataset_config, config, part_num=0):
        super().__init__(data_path, dataset_config, config=config)

        self.part_num = part_num

    def __getitem__(self, index):
        data = self.data[index]
        
        input_rgb = []

        img_meta = None

        sphere_dataloader = None

        if "input_images" in self.dataset_config.get("selection", ["input_images"]):
            with open(os.path.join(data["nuscenes"], "transforms", "transforms_ego.json"), "r") as f:
                input_data = json.load(f)

            input_rgb = []
            all_c2w = []
            K = np.zeros((3,4)) # (C,3,4)
            K[0,0] = input_data['fl_x']
            K[1,1] = input_data['fl_y']
            K[2,2] = 1
            K[0,2] = input_data['cx']
            K[1,2] = input_data['cy']

            for frame in input_data["frames"]:
                input_rgb.append(imread(os.path.join(data["nuscenes"], "transforms", frame["file_path"]), "unchanged")[:,:,:3].astype(np.float32))
                all_c2w.append(frame["transform_matrix"])


            input_rgb = self.transforms(input_rgb)

            img_shape = [img.shape for img in input_rgb]

            img_meta = dict(
                K=K,
                c2w=all_c2w,
                img_shape=img_shape,
            )
            

        if "sphere_dataset" in self.dataset_config.get("selection", ["sphere_dataset"]):
            if self.dataset_config.get("whole_image", False):
                filename = "train_dataset_"
            else:
                filename = "train_dataset_shuffled_"
            sphere_dataloader = []
            if self.dataset_config.get("whole_image", False):
                view_ids = random.sample(range(80), self.dataset_config.get("num_imgs",1))
            else:
                view_ids = np.arange(self.part_num, self.part_num + self.dataset_config.get("num_imgs",1))
            for view_id in view_ids:
                with open(os.path.join(data["sphere"], f"{filename}{view_id}.npy"), "rb") as f:
                    sphere_dataloader.append(np.load(f))
            sphere_dataloader = np.concatenate(sphere_dataloader)

        
        return (input_rgb, img_meta, sphere_dataloader)
    