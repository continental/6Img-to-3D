# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import os, sys
from datetime import datetime

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from rendering import render_rays

def visualize_dataset(dataset, H, W, n_max=-1, shuffle=False):
    rgb = dataset[:,6:9].numpy()
    rgb[dataset[:,9] == 0] /= 3
    rgb = rgb.reshape(-1,H,W,3)
    if n_max > len(dataset):
        n_max = len(dataset)
    if n_max < 0:
        n_max = len(dataset)

    if shuffle:
        np.random.shuffle(rgb)
    
    for i in range(n_max):
        plt.imshow(rgb[i])
        plt.show()
    

@torch.no_grad()
def test(model, hn, hf, dataset, chunk_size=10, img_indexes=None, nb_bins=192, H=400, W=400, device='cpu', folder='results', tqdm_disable=False, show=False, box_rendering=False):
    if img_indexes is None:
        img_indexes = np.arange(0, len(dataset) // (H * W))

    if isinstance(img_indexes, int) or isinstance(img_indexes, float):
        img_index = [img_indexes]

    folder = os.path.join(folder, datetime.today().strftime('%Y-%m-%d-%H-%M'))

    if not os.path.exists(folder):
        os.makedirs(folder)

    psnr_list = []
    
    pbar = tqdm(img_indexes, disable=tqdm_disable)
    for img_index in pbar:
        ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
        ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
        ground_truth_image = dataset[img_index * H * W: (img_index + 1) * H * W, 6:9].reshape(H, W, 3).numpy()
        mask = dataset[img_index * H * W: (img_index + 1) * H * W, 9].reshape(H, W).numpy()

        data = []
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins, box_rendering=box_rendering)
            data.append(regenerated_px_values)
        img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
        img = np.clip(img, 0, 1)

        ground_truth_image[mask == 0] /= 3

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[0].set_title('Generated Image')
        ax[1].imshow(ground_truth_image)
        ax[1].axis('off')
        ax[1].set_title('Ground Truth Image')
        plt.savefig(f'{folder}/{img_index}.png', bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close()

        psnr = cv2.PSNR(img[mask==1], ground_truth_image[mask==1], R=1)
        psnr_list.append(psnr)

        pbar.set_description(f'PSNR: {np.mean(psnr_list):.2f} dB')


    np.save(os.path.join(folder, 'psnr.npy'), np.array(psnr_list))
    return psnr_list


def train(model, optimizer, loss_fct, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5), nb_bins=192, visualize_every=0, tqdm_disable=False, box_rendering=False):
    training_loss = []
    for epoch in range(nb_epochs):
        epoch_training_loss = []
        pbar = tqdm(data_loader, disable=tqdm_disable)
        for n_batch, batch in enumerate(pbar):
            mask = batch[:,9].bool()
            
            if mask.sum() > 0:
                ray_origins = batch[mask, :3].to(device)
                ray_directions = batch[mask, 3:6].to(device)
                ground_truth_px_values = batch[mask, 6:9].to(device)
                
                regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, box_rendering=box_rendering)
                loss = loss_fct(regenerated_px_values, ground_truth_px_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_training_loss.append(loss.item())
                pbar.set_description(f'Epoch {epoch}/{nb_epochs} | Loss: {np.mean(epoch_training_loss):.4f}')

                if visualize_every > 0:
                    if n_batch % visualize_every == 0:
                        triplane = [[
                            model.xy_plane.detach(),
                            model.xz_plane.detach(),
                            model.yz_plane.detach()
                        ]]
                        
                        plot_avg_features_grayscale(triplane)

        scheduler.step()
        training_loss.append(epoch_training_loss)

    
    return training_loss

@torch.no_grad()
def visualize_loss(loss, mode='iter'):
    loss = np.array(loss)
    if len(loss.shape) == 1:
        loss = loss.reshape(1, -1)

    if mode == 'iter':
        for i in range(loss.shape[0]):
            plt.plot(np.arange(i * loss.shape[1], (i + 1) * loss.shape[1]), loss[i])

        plt.xlabel('iteration')
        plt.ylabel('Loss')

    elif mode == 'epoch':
        plt.plot(np.arange(loss.shape[0]), loss.mean(axis=1))
        plt.xlabel('epoch')
        plt.ylabel('Loss')

    plt.ylim(0, loss.max() * 1.1)
    plt.show()

@torch.no_grad()
def plot_avg_features_grayscale(triplane):
    
    N_multires = len(triplane)
    N_planes = len(triplane[0])

    # turn into np and avg.
    def get_avg_plane(matrix_torch):
        avg_tensor = torch.mean(matrix_torch, dim=2)
        return avg_tensor

    # Create a 6 x 3 subplot grid
    fig, axes = plt.subplots(nrows=N_multires, ncols=N_planes, figsize=(2 * N_planes, 2 * N_multires))
    cols = ['XY', 'XZ', 'YZ'] # x-axis labels
    axes = axes.reshape(N_multires, N_planes)
    for i, plane_res in enumerate(triplane):
        for j, plane in enumerate(plane_res):
            avg_plane = get_avg_plane(plane)
            axes[i, j].imshow(avg_plane.cpu().numpy(), cmap='gray')
            
    for i in range(N_planes):
        axes[0, i].set_title(cols[i])

    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    plt.show()