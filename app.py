import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
import av
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
from tools.util import all_file, load_mask_list, crop_img, pad_img, crop_human_clip_auto_context, get_mask, \
    refine_img_prepross
import gradio as gr
import json

MOTION_TRIGGER_WORD = {
    'sports_basketball_gym': [],
    'sports_nba_pass': [],
    'sports_nba_dunk': [],
    'movie_BruceLee1': [],
    'shorts_kungfu_match1': [],
    'shorts_kungfu_desert1': [],
    'parkour_climbing': [],
    'dance_indoor_1': [],
}
css_style = "#fixed_size_img {height: 500px;}"

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

    def run(self, ref_image_pil, template_name):

        template_dir = os.path.join(self.args.assets_dir, 'video_template')
        template_path = os.path.join(template_dir, template_name)
        template_info = self.load_template(template_path)

        target_fps = template_info['target_fps']
        video_path = template_info['video_path']
        pose_video_path = template_info['pose_video_path']
        bk_video_path = template_info['bk_video_path']
        occ_video_path = template_info['occ_video_path']

        # ref_image_pil = Image.open(ref_img_path).convert('RGB')
        source_image = np.array(ref_image_pil)
        source_image, mask = process_seg(source_image[..., ::-1])
        source_image = source_image[..., ::-1]
        source_image = crop_img(source_image, mask)
        source_image, _ = pad_img(source_image, [255, 255, 255])
        ref_image_pil = Image.fromarray(source_image)

        # load tgt
        vid_images = read_frames(video_path)
        if bk_video_path is None:
            n_frame = len(vid_images)
            tw, th = vid_images[0].size
            bk_images = init_bk(n_frame, tw, th)
        else:
            bk_images = read_frames(bk_video_path)

        if occ_video_path is not None:
            occ_mask_images = read_frames(occ_video_path)
            print('load occ from %s' % occ_video_path)
        else:
            occ_mask_images = None
            print('no occ masks')

        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)

        start_idx, end_idx = template_info['time_crop']['start_idx'], template_info['time_crop']['end_idx']
        start_idx = max(0, start_idx)
        end_idx = min(len(pose_images), end_idx)

        pose_images = pose_images[start_idx:end_idx]
        vid_images = vid_images[start_idx:end_idx]
        bk_images = bk_images[start_idx:end_idx]
        if occ_mask_images is not None:
            occ_mask_images = occ_mask_images[start_idx:end_idx]

        self.args.L = len(pose_images)
        max_n_frames = self.args.MAX_FRAME_NUM
        if self.args.L > max_n_frames:
            pose_images = pose_images[:max_n_frames]
            vid_images = vid_images[:max_n_frames]
            bk_images = bk_images[:max_n_frames]
            if occ_mask_images is not None:
                occ_mask_images = occ_mask_images[:max_n_frames]
            self.args.L = len(pose_images)

        bk_images_ori = bk_images.copy()
        vid_images_ori = vid_images.copy()

        overlay = 4
        pose_images, vid_images, bk_images, bbox_clip, context_list, bbox_clip_list = crop_human_clip_auto_context(
            pose_images, vid_images, bk_images, overlay)

        clip_pad_list_context = []
        clip_padv_list_context = []
        pose_list_context = []
        vid_bk_list_context = []
        for frame_idx in range(len(pose_images)):
            pose_image_pil = pose_images[frame_idx]
            pose_image = np.array(pose_image_pil)
            pose_image, _ = pad_img(pose_image, color=[0, 0, 0])
            pose_image_pil = Image.fromarray(pose_image)
            pose_list_context.append(pose_image_pil)

            vid_bk = bk_images[frame_idx]
            vid_bk = np.array(vid_bk)
            vid_bk, padding_v = pad_img(vid_bk, color=[255, 255, 255])
            pad_h, pad_w, _ = vid_bk.shape
            clip_pad_list_context.append([pad_h, pad_w])
            clip_padv_list_context.append(padding_v)
            vid_bk_list_context.append(Image.fromarray(vid_bk))

        print('start to infer...')
        video = self.pipe(
            ref_image_pil,
            pose_list_context,
            vid_bk_list_context,
            self.width,
            self.height,
            len(pose_list_context),
            self.args.steps,
            self.args.cfg,
            generator=self.generator,
        ).videos[0]

        # post-process video
        video_idx = 0
        res_images = [None for _ in range(self.args.L)]
        for k, context in enumerate(context_list):
            start_i = context[0]
            bbox = bbox_clip_list[k]
            for i in context:
                bk_image_pil_ori = bk_images_ori[i]
                vid_image_pil_ori = vid_images_ori[i]
                if occ_mask_images is not None:
                    occ_mask = occ_mask_images[i]
                else:
                    occ_mask = None

                canvas = Image.new("RGB", bk_image_pil_ori.size, "white")

                pad_h, pad_w = clip_pad_list_context[video_idx]
                padding_v = clip_padv_list_context[video_idx]

                image = video[:, video_idx, :, :].permute(1, 2, 0).cpu().numpy()
                res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
                res_image_pil = res_image_pil.resize((pad_w, pad_h))

                top, bottom, left, right = padding_v
                res_image_pil = res_image_pil.crop((left, top, pad_w - right, pad_h - bottom))

                w_min, w_max, h_min, h_max = bbox
                canvas.paste(res_image_pil, (w_min, h_min))

                mask_full = np.zeros((bk_image_pil_ori.size[1], bk_image_pil_ori.size[0]), dtype=np.float32)
                res_image = np.array(canvas)
                bk_image = np.array(bk_image_pil_ori)

                mask = get_mask(self.mask_list, bbox, bk_image_pil_ori)
                mask = cv2.resize(mask, res_image_pil.size, interpolation=cv2.INTER_AREA)
                mask_full[h_min:h_min + mask.shape[0], w_min:w_min + mask.shape[1]] = mask

                res_image = res_image * mask_full[:, :, np.newaxis] + bk_image * (1 - mask_full[:, :, np.newaxis])

                if occ_mask is not None:
                    vid_image = np.array(vid_image_pil_ori)
                    occ_mask = np.array(occ_mask)[:, :, 0].astype(np.uint8)  # [0,255]
                    occ_mask = occ_mask / 255.0
                    res_image = res_image * (1 - occ_mask[:, :, np.newaxis]) + vid_image * occ_mask[:, :,
                                                                                           np.newaxis]
                if res_images[i] is None:
                    res_images[i] = res_image
                else:
                    factor = (i - start_i + 1) / (overlay + 1)
                    res_images[i] = res_images[i] * (1 - factor) + res_image * factor
                res_images[i] = res_images[i].astype(np.uint8)

                video_idx = video_idx + 1
        return res_images


class WebApp():
    def __init__(self, debug_mode=False):
        self.args_base = {
            "device": "cuda",
            "output_dir": "output_demo",
            "img": None,
            "pos_prompt": '',
            "motion": "sports_basketball_gym",
            "motion_dir": "./assets/test_video_trunc",
        }

        self.args_input = {}  # for gr.components only
        self.gr_motion = list(MOTION_TRIGGER_WORD.keys())

        # fun fact: google analytics doesn't work in this space currently
        self.gtag = os.environ.get('GTag')

        self.ga_script = f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={self.gtag}"></script>
            """
        self.ga_load = f"""
            function() {{
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());

                gtag('config', '{self.gtag}');
            }}
            """

        # # pre-download base model for better user experience
        self.model = MIMO()

        self.debug_mode = debug_mode  # turn off clip interrogator when debugging for faster building speed

    def title(self):

        gr.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <a href="https://menyifang.github.io/projects/En3D/index.html" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
            </a>
            <div>
                <h1 >MIMO Demo</h1>

                </div>
            </div>
            </div>
            """
        )

    def get_template(self, num_cols=3):
        self.args_input['motion'] = gr.State('sports_basketball_gym')
        num_cols = 2
        lora_gallery = gr.Gallery(label='Gallery', columns=num_cols, height=500,
                                  value=[(os.path.join(self.args_base['motion_dir'], f"{motion}.mp4"), '') for
                                         motion in
                                         self.gr_motion],
                                  show_label=True,
                                  selected_index=0)

        lora_gallery.select(self._update_selection, inputs=[], outputs=[self.args_input['motion']])
        print(self.args_input['motion'])

    def _update_selection(self, selected_state: gr.SelectData):
        return self.gr_motion[selected_state.index]

    def run_process(self, *values):
        gr_args = self.args_base.copy()
        print(self.args_input.keys())
        for k, v in zip(list(self.args_input.keys()), values):
            gr_args[k] = v

        ref_image_pil = gr_args['img']  # pil image

        template_name = gr_args['motion']
        print('template_name:', template_name)

        save_dir = 'output'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # generate uuid
        case = datetime.now().strftime("%Y%m%d%H%M%S")
        outpath = f"{save_dir}/{case}.mp4"

        res = self.model.run(ref_image_pil, template_name)
        imageio.mimsave(outpath, res, fps=30, quality=8, macro_block_size=1)
        print('save to %s' % outpath)

        return outpath

    def preset_library(self):
        with gr.Blocks() as demo:
            with gr.Accordion(label="🧭 Guidance:", open=True, elem_id="accordion"):
                with gr.Row(equal_height=True):
                    gr.Markdown("""
                    - ⭐️ <b>step1：</b>Upload a character image or select one from the examples
                    - ⭐️ <b>step2：</b>Choose a motion template from the gallery
                    - ⭐️ <b>step3：</b>Click "Run" to generate the animation
                    - <b>Note: </b> The input character image should be full-body, front-facing, no occlusion, no handheld objects
                    """)

            with gr.Row():
                img_input = gr.Image(label='Input image', type="pil", elem_id="fixed_size_img")
                self.args_input['img'] = img_input

                with gr.Column():
                    self.get_template(num_cols=3)
                    submit_btn_load3d = gr.Button("Run", variant='primary')
                with gr.Column(scale=1.2):
                    res_vid = gr.Video(format="mp4", label="Generated Result", autoplay=True, elem_id="fixed_size_img")

            submit_btn_load3d.click(self.run_process,
                                    inputs=list(self.args_input.values()),
                                    outputs=[res_vid],
                                    scroll_to_output=True,
                                    )

            gr.Examples(examples=[
                ['./assets/test_image/sugar.jpg'],
                ['./assets/test_image/ouwen1.png'],
                ['./assets/test_image/actorhq_A1S1.png'],
                ['./assets/test_image/actorhq_A7S1.png'],
                ['./assets/test_image/cartoon1.png'],
                ['./assets/test_image/cartoon2.png'],
                ['./assets/test_image/sakura.png'],
                ['./assets/test_image/kakashi.png'],
                ['./assets/test_image/sasuke.png'],
                ['./assets/test_image/avatar.jpg'],
            ], inputs=[img_input],
                examples_per_page=20, label="Examples", elem_id="examples",
            )

    def ui(self):
        with gr.Blocks(css=css_style) as demo:
            self.title()
            self.preset_library()
            demo.load(None, js=self.ga_load)

        return demo


app = WebApp(debug_mode=False)
demo = app.ui()

if __name__ == "__main__":
    demo.queue(max_size=100)
    demo.launch(share=False)
