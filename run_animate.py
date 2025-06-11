import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_edit_bkfill import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long_edit_bkfill_roiclip import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames
import cv2
from tools.human_segmenter import human_segmenter
import imageio
from tools.util import all_file, load_mask_list, crop_img, pad_img, crop_human, init_bk
from tools.util import load_video_fixed_fps
import json

seg_path = './assets/matting_human.pb'
segmenter = human_segmenter(model_path=seg_path)


def process_seg(img):
    rgba = segmenter.run(img)
    mask = rgba[:, :, 3]
    color = rgba[:, :, :3]
    alpha = mask / 255
    bk = np.ones_like(color) * 255
    color = color * alpha[:, :, np.newaxis] + bk * (1 - alpha[:, :, np.newaxis])
    color = color.astype(np.uint8)
    return color, mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/prompts/animation_edit.yaml')
    parser.add_argument("-W", type=int, default=784)
    parser.add_argument("-H", type=int, default=784)
    parser.add_argument("-L", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--assets_dir", type=str, default='./assets')
    parser.add_argument("--ref_pad", type=int, default=1)
    parser.add_argument("--use_bk", type=int, default=1)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--MAX_FRAME_NUM", type=int, default=150)
    args = parser.parse_args()
    return args


class MIMO():
    def __init__(self, debug_mode=False):
        args = parse_args()

        config = OmegaConf.load(args.config)

        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

        vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to("cuda", dtype=weight_dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        inference_config_path = config.inference_config
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")

        pose_guider = PoseGuider(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device="cuda"
        )

        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            config.image_encoder_path
        ).to(dtype=weight_dtype, device="cuda")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        self.generator = torch.manual_seed(args.seed)

        self.width, self.height = args.W, args.H

        # load pretrained weights
        denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )
        pose_guider.load_state_dict(
            torch.load(config.pose_guider_path, map_location="cpu"),
        )

        self.pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.pipe = self.pipe.to("cuda", dtype=weight_dtype)

        self.args = args

        # load mask
        mask_path = os.path.join(self.args.assets_dir, 'masks', 'alpha2.png')
        self.mask_list = load_mask_list(mask_path)

    def load_template(self, template_path):
        video_path = os.path.join(template_path, 'vid.mp4')
        pose_video_path = os.path.join(template_path, 'sdc.mp4')
        bk_video_path = os.path.join(template_path, 'bk.mp4')
        occ_video_path = os.path.join(template_path, 'occ.mp4')
        if not os.path.exists(occ_video_path):
            occ_video_path = None
        config_file = os.path.join(template_path, 'config.json')
        with open(config_file) as f:
            template_data = json.load(f)
        template_info = {}
        template_info['video_path'] = video_path
        template_info['pose_video_path'] = pose_video_path
        template_info['bk_video_path'] = bk_video_path
        template_info['occ_video_path'] = occ_video_path
        template_info['target_fps'] = template_data['fps']
        template_info['time_crop'] = template_data['time_crop']
        template_info['frame_crop'] = template_data['frame_crop']
        template_info['layer_recover'] = template_data['layer_recover']
        return template_info

    def run(self, ref_img_path, template_path):

        template_name = os.path.basename(template_path)
        # template_info = self.load_template(template_path)

        target_fps = 30
        video_path = os.path.join(template_path, 'sdc.mp4')
        pose_video_path = os.path.join(template_path, 'sdc.mp4')
        bk_video_path = None

        ref_image_pil = Image.open(ref_img_path).convert('RGB')
        source_image = np.array(ref_image_pil)
        source_image, mask = process_seg(source_image[..., ::-1])
        source_image = source_image[..., ::-1]
        source_image = crop_img(source_image, mask)
        source_image, _ = pad_img(source_image, [255, 255, 255])
        ref_image_pil = Image.fromarray(source_image)

        # load tgt
        vid_bk_list = []
        vid_images = load_video_fixed_fps(video_path, target_fps=target_fps)

        if bk_video_path is None:
            n_frame = len(vid_images)
            tw, th = vid_images[0].size
            bk_images = init_bk(n_frame, tw, th)
        else:
            bk_images = load_video_fixed_fps(bk_video_path, target_fps=target_fps)

        pose_list = []
        pose_images = load_video_fixed_fps(pose_video_path, target_fps=target_fps)

        self.args.L = len(pose_images)
        max_n_frames = self.args.MAX_FRAME_NUM
        if self.args.L > max_n_frames:
            pose_images = pose_images[:max_n_frames]
            vid_images = vid_images[:max_n_frames]
            bk_images = bk_images[:max_n_frames]
            self.args.L = len(pose_images)

        # crop pose with human-center
        pose_images, vid_images, bk_images = crop_human(pose_images, vid_images, bk_images)

        for frame_idx in range(len(pose_images)):
            pose_image_pil = pose_images[frame_idx]
            pose_image = np.array(pose_image_pil)
            pose_image, _ = pad_img(pose_image, color=[0, 0, 0])
            pose_image_pil = Image.fromarray(pose_image)
            pose_list.append(pose_image_pil)  # for infer, 3072, 1024)

            vid_bk = bk_images[frame_idx]
            vid_bk = np.array(vid_bk)
            vid_bk, _ = pad_img(vid_bk, color=[255, 255, 255])
            vid_bk_list.append(Image.fromarray(vid_bk))

        print('start to infer...')
        video = self.pipe(
            ref_image_pil,
            pose_list,
            vid_bk_list,
            self.width,
            self.height,
            len(pose_images),
            self.args.steps,
            self.args.cfg,
            generator=self.generator,
        ).videos[0]

        res_images = []
        for video_idx in range(len(pose_images)):
            image = video[:, video_idx, :, :].permute(1, 2, 0).cpu().numpy()
            res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
            res_images.append(res_image_pil)

        return res_images, target_fps


def main():
    model = MIMO()

    ref_img_path = './assets/test_image/actorhq_A7S1.png'

    template_path = './assets/video_template/syn_basketball_06_13'

    save_dir = 'output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('refer_img: %s' % ref_img_path)
    print('template_vid: %s' % template_path)

    ref_name = os.path.basename(ref_img_path).split('.')[0]
    template_name = os.path.basename(template_path)
    outpath = f"{save_dir}/{template_name}_{ref_name}.mp4"

    res, target_fps = model.run(ref_img_path, template_path)
    imageio.mimsave(outpath, res, fps=target_fps, quality=8, macro_block_size=1)
    print('save to %s' % outpath)


if __name__ == "__main__":
    main()
