# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ============================================================================== 

import argparse
import os
import warnings

import numpy as np
import torch
import json
import imageio
from mmcv import Config
from tqdm import tqdm

warnings.filterwarnings("ignore")
import cv2
import lpips
from pytorch_msssim import ssim
import time
import pickle

from builder import data_builder, model_builder

from triplane_decoder.rendering import render_rays
from triplane_decoder.decoder import TriplaneDecoder
from triplane_decoder.pif import PIF

from PIL import Image

import pandas as pd



def main(local_rank, args):
    # global settings
    logdir = f'eval/{time.strftime("%b%d_%H-%M-%S", time.localtime())}_{args.log_dir}'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(os.path.join(logdir,"imgs")):
        os.makedirs(os.path.join(logdir,"imgs"))

    if args.img_gt and not os.path.exists(os.path.join(logdir,"imgs_gt")):
        os.makedirs(os.path.join(logdir,"imgs_gt"))

    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)

    if args.dataset_config != "":
        dataset_config = Config.fromfile(args.dataset_config)
        cfg["dataset_params"] = dataset_config["dataset_params"]

    Config.dump(cfg, os.path.join(logdir, "tpv_config.py"))

    triplane_encoder = model_builder.build(cfg.model)
    triplane_decoder = TriplaneDecoder(cfg)

    if cfg.pif:
        path = cfg.pif_transforms
        with open(os.path.join(path,"transforms/transforms_ego.json"), "r") as f:
            transforms = json.load(f)
        M_cameras = []
        M_cameras += [torch.tensor(frame["transform_matrix"]) for frame in transforms["frames"]]
        M_cameras = torch.stack(M_cameras)

        imgs = [torch.from_numpy(imageio.imread(os.path.join(path,"images",f"{i}_rgb.png")))[...,:3] for i in range(len(M_cameras))]
                        #  (f"{i}_rgb.png").convert("RGB")) for i in range(len(M_cameras))]
        imgs = torch.stack(imgs).float().permute(0,3,1,2)

        fl_x = transforms['fl_x']
        fl_y = transforms['fl_y']
        cx = transforms['cx']
        cy = transforms['cy']
        image_width = transforms['w']
        image_height = transforms['h']

        pif = PIF(
            focal_length=torch.tensor([fl_x, fl_y]),
            principal_point=torch.tensor([cx, cy]),
            image_size=torch.tensor([image_height, image_width]),
            c2w=M_cameras
        )
    else:
        pif = None

    

    triplane_decoder = triplane_decoder.cuda()
    triplane_encoder = triplane_encoder.cuda()
    if pif is not None:
        pif = pif.cuda()

    print('done building models')


    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location='cpu')

        triplane_decoder_ckpt = ckpt['ttp']
        if 'module.full_net.params' in triplane_decoder_ckpt and 'module.decoder_net.params' not in triplane_decoder_ckpt:
            triplane_decoder_ckpt['module.decoder_net.params'] = triplane_decoder_ckpt['module.full_net.params'] 
        triplane_decoder_ckpt = {k.replace("module.",""): v for k,v in triplane_decoder_ckpt.items()}
        print(triplane_decoder.load_state_dict(triplane_decoder_ckpt, strict=False))
        print("loaded triplane decoder weights")

        
        triplane_generator_ckpt = ckpt['tpv']
        triplane_generator_ckpt = {k.replace("module.",""): v for k,v in triplane_generator_ckpt.items()}
        print(triplane_encoder.load_state_dict(triplane_generator_ckpt, strict=False))
        print("loaded triplane encoder weights")

    _, val_dataset_loader = data_builder.build(cfg)

    lpips_loss_fct = lpips.LPIPS(net='alex').cuda()       

    triplane_decoder.eval()
    triplane_encoder.eval()

    results = []

    if args.time:
        t_encode = []
        t_decode = []

    with torch.no_grad():
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

            for scene_index, (imgs, img_metas, val_dataset) in tqdm(enumerate(val_dataset_loader), total=len(val_dataset_loader)):

                
                val_dataset = val_dataset[0].dataset
                W, H = val_dataset.intrinsics.width, val_dataset.intrinsics.height
                val_dataset = val_dataset.dataset
                
                if args.gif:
                    gif = []
                    depth_gif = []
                if args.gif_gt:
                    gif_gt = []
                    if (val_dataset.size(1) == 11) and args.depth :
                        depth_gif_gt = []


                imgs = imgs.cuda()

                if args.time:
                    t0_encode= time.time()

                triplane, features = triplane_encoder(img=imgs, img_metas=img_metas)

                if args.time:
                    torch.cuda.current_stream().synchronize()
                    t_encode.append(time.time() - t0_encode)


                triplane_decoder.update_planes(triplane)
                if pif is not None:
                    pif.update_imgs(features[0])

                if args.num_img >0:
                    index_set = range(min(len(val_dataset) // (H * W), args.num_img))
                else :
                    index_set = range(len(val_dataset) // (H * W))

                for img_index in index_set:
                    ray_origins = val_dataset[img_index * H * W: (img_index + 1) * H * W, :3].cuda()
                    ray_directions = val_dataset[img_index * H * W: (img_index + 1) * H * W, 3:6].cuda()
                    ground_truth_image = val_dataset[img_index * H * W: (img_index + 1) * H * W, 6:9].reshape(H, W, 3).cuda()
                    mask = val_dataset[img_index * H * W: (img_index + 1) * H * W, 9].reshape(H, W).numpy()

                    if args.depth:
                        depth_data = []

                        if val_dataset.size(1) == 11:
                            depth_gt = val_dataset[img_index * H * W: (img_index + 1) * H * W, 10].reshape(H, W).numpy()
                        else:
                            depth_gt = None

                    data = []
                    if args.time:
                        t0_decode = time.time()

                    for i in range(int(np.ceil(H / cfg.decoder.testing_batch_size))):
                        ray_origins_ = ray_origins[i * W * cfg.decoder.testing_batch_size : (i + 1) * W * cfg.decoder.testing_batch_size]
                        ray_directions_ = ray_directions[i * W * cfg.decoder.testing_batch_size: (i + 1) * W * cfg.decoder.testing_batch_size]
                        regenerated_px_values, _, depth= render_rays(triplane_decoder, ray_origins_, ray_directions_, cfg, pif=pif, training=False, only_coarse=args.single_sampling)

                        
                        data.append(regenerated_px_values)
                        if args.depth:
                            depth_data.append(depth)

                    if args.time:
                        torch.cuda.current_stream().synchronize()
                        t_decode.append(time.time() - t0_decode)

                    img = torch.cat(data).reshape(H, W, 3)
                    img = torch.clip(img, 0, 1)

                    if args.depth:
                        depth = torch.sqrt(torch.cat(depth_data).reshape(H,W) / 60)
                        depth = cv2.applyColorMap((255 * depth.cpu()).numpy().astype(np.uint8), cv2.COLORMAP_JET)

                        if depth_gt is not None:
                            depth_gt = np.clip(np.sqrt(depth_gt/60), 0, 1)
                            depth_gt = cv2.applyColorMap((255 * depth_gt).astype(np.uint8), cv2.COLORMAP_JET)


                    if cfg.decoder.whiteout:
                        ground_truth_image[mask == 0] = 1
                    
                    
                    lpips_metric = torch.mean(lpips_loss_fct(img.view(1,H,W,3).permute(0,3,1,2) * 2 - 1, ground_truth_image.view(1,H,W,3).permute(0,3,1,2) * 2 - 1)).item()
                    psnr_metric = cv2.PSNR(img.cpu().numpy(),                   ground_truth_image.cpu().numpy(), R=1)
                    ssim_metric = ssim(img.view(1,H,W,3), ground_truth_image.view(1,H,W,3), data_range=1, size_average=True).item()

                    if args.gif:
                        gif.append(Image.fromarray((255*img).view(H,W,3).cpu().numpy().astype(np.uint8)))
                        if args.depth:
                            depth_gif.append(Image.fromarray(depth))
                    else:
                        imageio.imwrite(os.path.join(logdir,"imgs", f"{scene_index}_{img_index}.png"),
                                        (255*img).cpu().numpy().astype(np.uint8))
                        if args.depth:
                            imageio.imwrite(os.path.join(logdir,"imgs", f"{scene_index}_{img_index}_depth.png"), depth)
                        if args.img_gt:
                            imageio.imwrite(os.path.join(logdir,"imgs_gt", f"{scene_index}_{img_index}.png"),
                                        (255*ground_truth_image).cpu().numpy().astype(np.uint8))
                            if args.depth and depth_gt is not None  :
                                imageio.imwrite(os.path.join(logdir,"imgs", f"{scene_index}_{img_index}gt_depth.png"), depth_gt)
                    if args.gif_gt:
                        gif_gt.append(Image.fromarray((255*ground_truth_image).view(H,W,3).cpu().numpy().astype(np.uint8)))
                        if args.depth and depth_gt is not None:
                            depth_gif_gt.append(Image.fromarray(depth_gt))
                    results.append({
                        "scene":scene_index,
                        "img": img_index,
                        "psnr":psnr_metric,
                        "lpips":lpips_metric,
                        "ssim": ssim_metric
                    })
                if args.gif:
                    gif[0].save(os.path.join(logdir,"imgs", f"{scene_index}.gif"), save_all=True, append_images=gif[1:], duration=50, loop=0) 
                    if args.depth:
                        depth_gif[0].save(os.path.join(logdir,"imgs", f"{scene_index}_depth.gif"), save_all=True, append_images=depth_gif[1:], duration=50, loop=0)                
                if args.gif_gt:
                    gif_gt[0].save(os.path.join(logdir,"imgs", f"{scene_index}gd.gif"), save_all=True, append_images=gif_gt[1:], duration=50, loop=0)
                    if args.depth:
                        depth_gif_gt[0].save(os.path.join(logdir,"imgs", f"{scene_index}gd_depth.gif"), save_all=True, append_images=depth_gif_gt[1:], duration=50, loop=0)


            df = pd.DataFrame(results)

            print(df[["psnr","lpips","ssim"]].mean())  

            df.to_csv(os.path.join(logdir,"metrics.csv"))


            if args.time:
                np.savetxt(os.path.join(logdir,'t_decode.txt'), t_decode)
                np.savetxt(os.path.join(logdir,'t_encode.txt'), t_encode)

                print(f"Median time encode: {np.median(t_encode)}")
                print(f"Median time decode: {np.median(t_decode)}")

                
        

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/config.py')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--gif", action='store_true')
    parser.add_argument("--gif-gt", action='store_true')
    parser.add_argument("--img-gt", action='store_true')
    parser.add_argument("--time", action='store_true')
    parser.add_argument("--depth", action='store_true')
    parser.add_argument("--single-sampling", action='store_true')
    parser.add_argument("--num-img", type=int, default=-1)
    parser.add_argument("--dataset-config", type=str, default="")
    


    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    ngpus = 0
    args.gpus = ngpus
    print(args)

    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    main(0, args)