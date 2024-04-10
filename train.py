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
import matplotlib
import lpips
import time

from builder import data_builder, model_builder

from triplane_decoder.lr_scheduling import get_cosine_schedule_with_warmup
from triplane_decoder.rendering import render_rays
from triplane_decoder.decoder import TriplaneDecoder
from triplane_decoder.losses import compute_tv_loss
from triplane_decoder.pif import PIF


matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter


def main(local_rank, args):
    # global settings


    if args.log_dir == "":
        writer = SummaryWriter()
    else:
        logdir = f'runs/{time.strftime("%b%d_%H-%M-%S", time.localtime())}_{args.log_dir}'
        writer = SummaryWriter(log_dir=logdir)

    
    save_dir = os.path.join(writer.logdir, 'models')

    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    
    Config.dump(cfg, os.path.join(writer.logdir, "config.py"))

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


    # resume and load
    if args.ckpt_path:
        assert os.path.isfile(args.ckpt_path)
        cfg.resume_from = args.ckpt_path
        print('ckpt path:', cfg.resume_from)
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if 'model' in ckpt:
            ckpt = ckpt['model']
        tpv_keys = triplane_encoder.state_dict().keys()
        ckpt = {k.replace("module.",""): v for k,v in ckpt.items()}
        # module_key = [key for key in ckpt.keys() if 'module.' in key]
        # if len(module_key) > 0:
        #     ckpt = revise_ckpt(ckpt)
        try:
            print(triplane_encoder.load_state_dict(ckpt, strict=False))
            print(f'successfully loaded ckpt')
        except Exception as e:
            print(e)

    # torch.cuda.empty_cache()
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location='cpu')
        try:
            triplane_decoder_ckpt = ckpt['ttp']
            if 'module.full_net.params' in triplane_decoder_ckpt and 'module.decoder_net.params' not in triplane_decoder_ckpt:
                triplane_decoder_ckpt['module.decoder_net.params'] = triplane_decoder_ckpt['module.full_net.params'] 
            triplane_decoder_ckpt = {k.replace("module.",""): v for k,v in triplane_decoder_ckpt.items()}
            print(triplane_decoder.load_state_dict(triplane_decoder_ckpt, strict=False))
        except:
            print('no ttp in ckpt')
        
        triplane_generator_ckpt = ckpt['tpv']
        triplane_generator_ckpt = {k.replace("module.",""): v for k,v in triplane_generator_ckpt.items()}

        print(triplane_encoder.load_state_dict(triplane_generator_ckpt, strict=False))
    
    
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(cfg)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.AdamW(list(triplane_encoder.parameters()) + list(triplane_decoder.parameters()), lr=cfg.optimizer.lr )#5e5 for training, 5e-6 for lpips finetuning
    mse_loss_fct = torch.nn.MSELoss()      
    lpips_loss_fct = lpips.LPIPS(net='vgg').cuda()       

    num_steps = len(train_dataset_loader) * cfg.optimizer.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.optimizer.num_training_steps,
            num_training_steps=num_steps,
        )
        

    if args.from_epoch > 0:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])  

        scheduler.step(args.from_epoch * len(train_dataset_loader))

    triplane_decoder.train()
    triplane_encoder.train()


    best_psnr = 0
    best_lpips = 1

    num_imgs = cfg.dataset_params.train_data_loader.get("num_imgs",1)

    if args.from_epoch > 0:
        start = args.from_epoch
    else:
        start = 0

    for epoch in range(start, num_steps // len(train_dataset_loader)):
        train_dataset_loader.part_num = (epoch * num_imgs) % (80 // num_imgs)
        try:
            triplane_decoder.train()
            triplane_encoder.train()

            loss_dict = {}
            loss_dict['loss'] = 0
            loss_dict['mse_loss'] = 0
            if cfg.optimizer.tv_loss_weight > 0:
                loss_dict['tv_loss'] = 0
            if cfg.optimizer.dist_loss_weight > 0:
                loss_dict['dist_loss'] = 0
            if cfg.optimizer.lpips_loss_weight > 0:
                loss_dict['lpips_loss'] = 0
            if cfg.optimizer.depth_loss_weight > 0:
                loss_dict['depth_loss'] = 0

            print(f"step {epoch}/100")
            if args.num_scenes > 0:
                total_scenes = min(args.num_scenes, len(train_dataset_loader))
            else:
                total_scenes = len(train_dataset_loader)
                
            pbar = tqdm(enumerate(train_dataset_loader), total=total_scenes)
            for i_iter_val, (imgs, img_metas, batch) in pbar:
                if (args.num_scenes > 0) and i_iter_val > total_scenes:
                    continue
                batch = torch.from_numpy(batch[0])
                # batch = batch[0]

                imgs = imgs.cuda() 

                triplane, features = triplane_encoder(img=imgs, img_metas=img_metas)

                triplane_decoder.update_planes(triplane)
                if pif is not None:
                    pif.update_imgs(features[0])

                # train_step
                mask = batch[:,9].bool()

                if mask.sum() > 0:
                    batch = batch.cuda()
                    ray_origins = batch[:, :3]
                    ray_directions = batch[:, 3:6]
                    ground_truth_px_values = batch[:, 6:9]
                    if cfg.optimizer.depth_loss_weight > 0:
                        ground_truth_depth = batch[:,10:]

                    if cfg.decoder.whiteout:
                        ground_truth_px_values[~mask] = 1
                    
                    regenerated_px_values, dist_loss, depth = render_rays(triplane_decoder, ray_origins, ray_directions, cfg, pif=pif, training=True)

                    mse_loss = mse_loss_fct(regenerated_px_values, ground_truth_px_values)

                    tv_loss = cfg.optimizer.tv_loss_weight * compute_tv_loss(triplane_decoder) if cfg.optimizer.tv_loss_weight > 0 else 0

                    dist_loss = cfg.optimizer.dist_loss_weight * dist_loss if cfg.optimizer.dist_loss_weight > 0 else 0

                    if cfg.optimizer.lpips_loss_weight > 0:
                        lpips_loss = cfg.optimizer.lpips_loss_weight *  \
                            torch.mean(lpips_loss_fct(regenerated_px_values.view(-1,48,64,3).permute(0,3,1,2) * 2 - 1, 
                                                    ground_truth_px_values.view(-1,48,64,3).permute(0,3,1,2) * 2 - 1))
                    else:
                        lpips_loss = 0

                    depth_loss = cfg.optimizer.depth_loss_weight * mse_loss_fct(torch.sqrt(depth/60), torch.sqrt(torch.clip(ground_truth_depth/60, 0,1))) if cfg.optimizer.depth_loss_weight > 0 else 0

                    loss = mse_loss + tv_loss + dist_loss + lpips_loss + depth_loss

                    if loss.isnan():
                        print("Loss is NaN")
                        continue

                    optimizer.zero_grad()
                    if cfg.optimizer.clip_grad_norm > 0.:
                        torch.nn.utils.clip_grad_norm_(triplane_decoder.parameters(), cfg.optimizer.clip_grad_norm)
                        torch.nn.utils.clip_grad_norm_(triplane_encoder.parameters(), cfg.optimizer.clip_grad_norm)

                    loss.backward()
                    optimizer.step()
                    scheduler.step(epoch * len(train_dataset_loader) + i_iter_val)


                    if (i_iter_val % 100) == 0: 

                        loss_dict['loss'] += loss.item()
                        loss_dict['mse_loss'] += mse_loss.item()
                        if cfg.optimizer.tv_loss_weight > 0:
                            loss_dict['tv_loss'] += tv_loss.item()
                        if cfg.optimizer.dist_loss_weight > 0: 
                            loss_dict['dist_loss'] += dist_loss.item()
                        if cfg.optimizer.lpips_loss_weight:
                            loss_dict['lpips_loss'] += lpips_loss.item()
                        if cfg.optimizer.depth_loss_weight > 0:
                            loss_dict['depth_loss'] += depth_loss.item()

                        pbar.set_description(f"loss: {loss.item():.4f}")
            for key in loss_dict.keys():
                loss_dict[key] /= (len(train_dataset_loader)/100)

            writer.add_scalars('Loss/train', loss_dict, epoch) 


            # save models 
            if epoch % 10 == 0:
                torch.save({
                    "ttp": triplane_decoder.state_dict(),
                    "tpv": triplane_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, os.path.join(save_dir, f"model_{epoch}.pth"))
            torch.save({
                "ttp": triplane_decoder.state_dict(),
                "tpv": triplane_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(save_dir, f"model_latest.pth"))

            optimizer.zero_grad()

            with torch.no_grad():
                    psnr_list = []
                    lpips_list = []

                    for i_iter_val, (imgs, img_metas, val_dataset) in enumerate(val_dataset_loader):
                        
                        val_dataset = val_dataset[0].dataset
                        W, H = val_dataset.intrinsics.width, val_dataset.intrinsics.height

                        triplane_decoder.eval()
                        triplane_encoder.eval()

                        imgs = imgs.cuda()

                        triplane, features = triplane_encoder(img=imgs, img_metas=img_metas)

                        triplane_decoder.update_planes(triplane)
                        if pif is not None:
                            pif.update_imgs(features[0])


                        for img_index in np.arange(0, len(val_dataset) // (H * W)):
                            if img_index == 19:
                                break
                            ray_origins = val_dataset[img_index * H * W: (img_index + 1) * H * W, :3].cuda()
                            ray_directions = val_dataset[img_index * H * W: (img_index + 1) * H * W, 3:6].cuda()
                            ground_truth_image = val_dataset[img_index * H * W: (img_index + 1) * H * W, 6:9].reshape(H, W, 3).cuda()
                            mask = val_dataset[img_index * H * W: (img_index + 1) * H * W, 9].reshape(H, W).numpy()

                            data = []
                            
                            for i in range(int(np.ceil(H / cfg.decoder.testing_batch_size))):
                                ray_origins_ = ray_origins[i * W * cfg.decoder.testing_batch_size : (i + 1) * W * cfg.decoder.testing_batch_size]
                                ray_directions_ = ray_directions[i * W * cfg.decoder.testing_batch_size: (i + 1) * W * cfg.decoder.testing_batch_size]
                                regenerated_px_values, dist_loss, _ = render_rays(triplane_decoder, ray_origins_, ray_directions_, cfg, pif=pif, training=False)
                                data.append(regenerated_px_values)

                            img = torch.cat(data).reshape(H, W, 3)
                            img = torch.clip(img, 0, 1)

                            if cfg.decoder.whiteout:
                                ground_truth_image[mask == 0] = 1
        
                            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                            ax[0].imshow(img.cpu())
                            ax[0].axis('off')
                            ax[0].set_title('Generated Image')
                            ax[1].imshow(ground_truth_image.cpu())
                            ax[1].axis('off')
                            ax[1].set_title('Ground Truth Image')               

                            writer.add_figure(f'Image{epoch}/{i_iter_val}', fig, global_step=img_index)
                            plt.close()

                            
                            lpips_metric = torch.mean(lpips_loss_fct(img.view(1,H,W,3).permute(0,3,1,2) * 2 - 1, 
                                                        ground_truth_image.view(1,H,W,3).permute(0,3,1,2) * 2 - 1)).item()
                            psnr = cv2.PSNR(img.cpu().numpy(), ground_truth_image.cpu().numpy(), R=1)
                            psnr_list.append(psnr)
                            lpips_list.append(lpips_metric)

                        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
                        imgs = torch.clip(imgs[0].detach().cpu() / 255 + 0.5,0,1).permute(0,2,3,1)
                        ax[0,0].imshow((imgs[2][:,:,[2,1,0]]))
                        ax[0,0].axis('off')
                        ax[0,1].imshow((imgs[0][:,:,[2,1,0]]))
                        ax[0,1].axis('off')
                        ax[0,2].imshow((imgs[1][:,:,[2,1,0]]))
                        ax[0,2].axis('off')
                        ax[1,0].imshow((imgs[4][:,:,[2,1,0]]))
                        ax[1,0].axis('off')
                        ax[1,1].imshow((imgs[3][:,:,[2,1,0]]))
                        ax[1,1].axis('off')
                        ax[1,2].imshow((imgs[5][:,:,[2,1,0]]))
                        ax[1,2].axis('off')

                        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)

                        writer.add_figure(f'Image{epoch}/{i_iter_val}', fig, global_step=img_index )

                    writer.add_scalar('val/psnr', np.mean(psnr_list), epoch)
                    writer.add_scalar('val/lpips', np.mean(lpips_list), epoch)
                    print(f"{args.log_dir} PSNR : {np.mean(psnr_list):.2f}, LPIPS : {np.mean(lpips_list):.2f}")

                    if np.mean(psnr_list) > best_psnr:
                        torch.save({
                            "ttp": triplane_decoder.state_dict(),
                            "tpv": triplane_encoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            }, os.path.join(save_dir, f"model_best_psnr.pth"))
                        best_psnr = np.mean(psnr_list)
                    if np.mean(lpips_list) < best_lpips:
                        torch.save({
                            "ttp": triplane_decoder.state_dict(),
                            "tpv": triplane_encoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            }, os.path.join(save_dir, f"model_best_lpips.pth"))
                        best_lpips = np.mean(lpips_list)                       


        except RuntimeError as e:
            print(e)

            torch.cuda.empty_cache()
            ckpt = torch.load(os.path.join(save_dir, f"model_latest.pth"), map_location='cpu')
            triplane_decoder.load_state_dict(ckpt['ttp'])
            triplane_encoder.load_state_dict(ckpt['tpv'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])


            

        

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/config.py')
    parser.add_argument('--ckpt-path', type=str, default='')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--num-scenes", type=int, default=-1)
    parser.add_argument("--from-epoch", type=int, default="-1")
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    ngpus = 0
    args.gpus = ngpus
    print(args)

    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    main(0, args)