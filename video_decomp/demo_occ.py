import shutil
import os
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
import sys

# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sam2_yy'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'segment-anything-2-main'))

# from sam import SamAutomaticMaskGenerator
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# from segment_anything_hq import sam_model_registry as hqsam_model_registry, SamPredictor as HqSamPredictor
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
# from hi_sam.modeling.build import model_registry as hisam_model_registry
# from hi_sam.modeling.predictor import SamPredictor as HiSamPredictor
# from mask_painter import mask_painter, add_color, show_automask_anno
from tools.painter import mask_painter, show_automask_anno
from tools.util import clean_mask
import os, imageio, glob
from pathlib import Path
# from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig  # install detectron2 first
from depth_anything_v2.dpt import DepthAnythingV2
# from sam2.utils.util import boxes_from_bitmap, is_rectangle_intersect
import onnxruntime as ort
from tools.auto_mask import masks_update, masks_update_single
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


sam_models = {
    'vit_h': './models/sam_vit_h_4b8939.pth',
    'vit_l': './models/sam_vit_l_0b3195.pth',
    'vit_b': './checkpoints/sam_vit_b_01ec64.pth'
}

hqsam_models = {
    'vit_h': './checkpoints/sam_hq_vit_h.pth',
    'vit_l': './checkpoints/sam_hq_vit_l.pth',
    'vit_b': './checkpoints/sam_hq_vit_b.pth',
}

sam2_models = {
    'hiera_l': './models/sam2_hiera_large.pt',
    'hiera_b': './checkpoints/sam2_hiera_base_plus.pt',
    'hiera_s': './checkpoints/sam2_hiera_small.pt',
    'hiera_t': './checkpoints/sam2_hiera_tiny.pt'
}

depth_models = {
    'vitl': './models/depth_anything_v2_vitl.pth',
    'vitb': './checkpoints/depth_anything_v2_vitb.pth',
    'vits': './checkpoints/depth_anything_v2_vits.pth'
}

depth_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class OccTracker:
    def __init__(self,
                 sam2_model_cfg='sam2_hiera_l.yaml',
                 sam2_model_type='hiera_l',
                 depth_model_type='vitl',
                 # detect_cfg_path,
                 # detect_ckpt_path,
                 # dbnet_ckpt_path,
                 grayscale=False,
                 overlap_threshold=0.4,
                 detect_text=False,
                 device='cuda:0'):
        print("start init models...")
        # initialise models
        self.grayscale = grayscale
        self.overlap_threshold = overlap_threshold
        self.detect_text = detect_text

        sam2 = build_sam2(sam2_model_cfg, sam2_models[sam2_model_type], device=device)
        self.predictor_sam = SAM2ImagePredictor(sam2)

        sam_model_type = 'vit_h'
        sam = sam_model_registry[sam_model_type](checkpoint=sam_models[sam_model_type]).to('cuda')
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.7, 
            box_nms_thresh=0.7, 
            stability_score_thresh=0.85, 
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )

        # self.predictor_sam2 = build_sam2_camera_predictor(sam2_model_cfg, sam2_models[sam2_model_type], device=device)
        checkpoint = "models/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.predictor_sam2 = build_sam2_video_predictor(model_cfg, checkpoint)

        # self.mask_generator = SamAutomaticMaskGenerator(sam)

        self.depth_anything = DepthAnythingV2(**depth_configs[depth_model_type])
        self.depth_anything.load_state_dict(
            torch.load(depth_models[depth_model_type], map_location='cpu'))
        self.depth_anything = self.depth_anything.to(device).eval()

        print("init models done...")

    def set_image_sam(self, img):
        self.predictor_sam.set_image(img)

    def reset_image_sam(self):
        # self.predictor_sam.reset_image()
        self.predictor_sam.reset_predictor()

    def reset_video_sam2(self):
        self.predictor_sam2.reset_state()

    def detect_person(self, img_cv2):
        # Detect humans in image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0)
        if valid_idx.sum() == 0:
            return None
        valid_scores = det_instances.scores[valid_idx]
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()  # 2，4

        # # get the main person
        main_id = 0
        print(len(boxes))
        if len(boxes) >= 1:
            main_id = self.get_main_person(boxes, img_cv2)

            bbox = boxes[main_id].astype('int')
            print('person_bbox:', bbox)
            return np.array(bbox)
        else:
            return None

    def get_main_person(self, boxes, frame):
        # get the main person closest to the center
        ## bbox: x0, y0, x1, y1
        center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])  # (x, y)
        boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
        dists = np.linalg.norm(boxes_center - center, axis=1)
        main_id = np.argmin(dists)
        return main_id

    def pred_automask(self, image):
        """
        predict automask
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format

        """
        # h, w = image.shape[:2]
        # mask_all = self.mask_generator.generate(image, multimask_output=False)

        mask_all = self.mask_generator.generate(image)
        num_masks = len(mask_all)
        print("auto mask num:", num_masks)

        return mask_all

    def pred_automask_large(self, image):
        masks_l = self.mask_generator.generate(image)
        print("auto mask num:", len(masks_l))
        # masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        masks_l = masks_update_single(masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        print("auto mask num:", len(masks_l))
        return masks_l

    def pred_sam(self, input_box):

        masks, scores, logits = self.predictor_sam.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )  # 1,h,w

        # print("scores:{}".format(scores))
        best_index = np.argmax(scores)
        mask = masks[best_index].astype(np.uint8)  # [0,1]

        # mask_input = logits[np.argmax(scores), :, :]
        # masks, _, _ = self.predictor_sam.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     mask_input=mask_input[None, :, :],
        #     multimask_output=False,
        # )
        # print(masks.shape)
        # mask = masks[0].astype(np.uint8)

        return mask

    def pred_depth(self, img_cv2):
        # estimate depth
        depth = self.depth_anything.infer_image(img_cv2, 518)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)  # h,w
        return depth

    def pred_sam2_depth(self, frames, mask_list, seg_dir, dep_dir, occlude_dir):

        ann_frame_idx = 0

        # cap = cv2.VideoCapture(vid_path)
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print("video width: {}, height: {}, fps: {}".format(width, height, fps))

        if_init = False
        count = 0
        frames_occ = []

        for frame in frames:
            height, width, _ = frame.shape
            # while True:
            #     ret, frame_cv2 = cap.read()
            #     if not ret:
            #         break
            # frame = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)

            frame_cv2 = frame[..., ::-1]

            depth = self.pred_depth(frame_cv2)
            cv2.imwrite(os.path.join(dep_dir, '%07d.png' % (count)), depth)

            if not if_init:
                self.predictor_sam2.load_first_frame(frame)  # init state in this function
                if_init = True

                for i in range(len(mask_list)):
                    mask = mask_list[i]
                    # print('mask:', mask.shape)
                    # print(np.max(mask))
                    # print(np.min(mask))
                    # print(mask.dtype)

                    _, out_obj_ids, out_mask_logits = self.predictor_sam2.add_new_mask(
                        frame_idx=ann_frame_idx,
                        obj_id=i,
                        mask=mask,
                    )
                    # list of obj_ids

                mask0 = (out_mask_logits[0] > 0.0).cpu().numpy()
                mask0 = mask0[0].astype(np.uint8)
                if np.sum(mask0) > 0:
                    bbox0 = self.mask_find_bboxs(mask0)
                    # print(bbox0)
                    mask_bbox = np.zeros_like(mask0, dtype=np.uint8)
                    mask_bbox[bbox0[1]:bbox0[3], bbox0[0]:bbox0[2]] = 1

                avg_depth0 = self.avg_depth_value(depth, mask0)

                if self.grayscale:
                    mask_subject = mask0 * 255.
                    cv2.imwrite(os.path.join(seg_dir, '%07d.png' % (count)), mask_subject)
                else:
                    colored_mask = add_color(frame_cv2, mask0 * 255.)
                    mask_subject = colored_mask
                    cv2.imwrite(os.path.join(seg_dir, '%07d.png' % (count)), mask_subject)

                mask_concat = np.zeros((height, width), dtype=np.uint8)
                if np.sum(mask0) == 0 or len(out_obj_ids) <= 1:  # no person in this frame
                    mask_occlusion = mask_concat * 255.
                    cv2.imwrite(os.path.join(occlude_dir, '%07d.png' % (count)), mask_occlusion)
                else:
                    if self.detect_text:
                        text_bboxes, text_masked = self.detect_text_bbox(frame_cv2, bbox0, mask0)
                        if text_bboxes is not None:
                            self.predictor_hisam.set_image(text_masked)
                            _, hr_mask, _, _ = self.predictor_hisam.predict(multimask_output=False)
                            mask_text = hr_mask[0].astype(np.uint8)
                            mask_concat = cv2.bitwise_or(mask_concat, mask_text)
                            self.predictor_hisam.reset_image()

                    for j in range(1, len(out_obj_ids)):
                        mask = (out_mask_logits[j] > 0.0).cpu().numpy()
                        mask = mask[0].astype(np.uint8)

                        avg_depth = self.avg_depth_value(depth, mask)
                        intersect = cv2.bitwise_and(mask, mask_bbox)
                        if avg_depth > avg_depth0 and np.sum(intersect) > 0:
                            mask_concat = cv2.bitwise_or(mask_concat, mask)

                        # mask_concat = add_color(frame_cv2, mask_concat*255., flag=1)
                        if np.sum(mask_concat) > 0:
                            painted_mask = mask_painter(frame_cv2, mask_concat, background_alpha=0.75)
                        else:
                            painted_mask = mask_concat * 255.

                    if self.grayscale:
                        mask_occlusion = mask_concat * 255.
                        cv2.imwrite(os.path.join(occlude_dir, '%07d.png' % (count)), mask_occlusion)
                    else:
                        mask_occlusion = painted_mask
                        cv2.imwrite(os.path.join(occlude_dir, '%07d.png' % (count)), mask_occlusion)


            else:
                out_obj_ids, out_mask_logits = self.predictor_sam2.track(frame)

                mask0 = (out_mask_logits[0] > 0.0).cpu().numpy()
                mask0 = mask0[0].astype(np.uint8)
                if np.sum(mask0) > 0:
                    bbox0 = self.mask_find_bboxs(mask0)
                    # print(bbox0)
                    mask_bbox = np.zeros_like(mask0, dtype=np.uint8)
                    mask_bbox[bbox0[1]:bbox0[3], bbox0[0]:bbox0[2]] = 1

                avg_depth0 = self.avg_depth_value(depth, mask0)
                if self.grayscale:
                    mask_subject = mask0 * 255.
                    cv2.imwrite(os.path.join(seg_dir, '%07d.png' % (count)), mask_subject)
                else:
                    colored_mask = add_color(frame_cv2, mask0 * 255.)
                    mask_subject = colored_mask
                    cv2.imwrite(os.path.join(seg_dir, '%07d.png' % (count)), mask_subject)

                mask_concat = np.zeros((height, width), dtype=np.uint8)
                # print(count, np.sum(mask))
                if np.sum(mask0) == 0 or len(out_obj_ids) <= 1:  # no person in this frame
                    mask_occlusion = mask_concat * 255.
                    cv2.imwrite(os.path.join(occlude_dir, '%07d.png' % (count)), mask_occlusion)

                else:
                    if self.detect_text:
                        text_bboxes, text_masked = self.detect_text_bbox(frame_cv2, bbox0, mask0)
                        if text_bboxes is not None:
                            self.predictor_hisam.set_image(text_masked)
                            _, hr_mask, _, _ = self.predictor_hisam.predict(multimask_output=False)
                            mask_text = hr_mask[0].astype(np.uint8) * 255
                            mask_concat = cv2.bitwise_or(mask_concat, mask_text)
                            self.predictor_hisam.reset_image()

                    for j in range(1, len(out_obj_ids)):
                        mask = (out_mask_logits[j] > 0.0).cpu().numpy()
                        mask = mask[0].astype(np.uint8)
                        avg_depth = self.avg_depth_value(depth, mask)
                        intersect = cv2.bitwise_and(mask, mask_bbox)
                        if avg_depth > avg_depth0 and np.sum(intersect) > 0:
                            mask_concat = cv2.bitwise_or(mask_concat, mask)

                        # mask_concat = add_color(frame_cv2, mask_concat * 255., flag=1)
                        if np.sum(mask_concat) > 0 and np.sum(intersect) > 0:
                            painted_mask = mask_painter(frame_cv2, mask_concat, background_alpha=0.75)
                        else:
                            painted_mask = mask_concat * 255.

                    if self.grayscale:
                        mask_occlusion = mask_concat * 255.
                        cv2.imwrite(os.path.join(occlude_dir, '%07d.png' % (count)), mask_occlusion)
                    else:
                        mask_occlusion = painted_mask
                        cv2.imwrite(os.path.join(occlude_dir, '%07d.png' % (count)), mask_occlusion)

            frames_occ.append(mask_occlusion.astype(np.uint8))

            mask_show = np.hstack([self.convert2rgb(mask_subject), self.convert2rgb(mask_occlusion)])
            cv2.imwrite(os.path.join(seg_dir, '%07d.png' % (count)), mask_show)

            count += 1
        # cap.release()

        return frames_occ

    def judge_obj_valid(self, obj_mask, cur_mask):
        is_valid = 1
        num_obj = np.sum(obj_mask > 0)
        num_cur = np.sum(cur_mask > 0)
        # main object layer or not
        intersect = cv2.bitwise_and(obj_mask, cur_mask)
        num_inter = np.sum(intersect > 0)
        # print('num_obj:', num_obj)
        # print('num_cur:', num_cur)
        # print('num_inter:', num_inter)

        # print('inter ratio:', num_inter/num_cur)

        # if (num_inter > num_obj*0.8):
        if (num_inter > num_obj*0.8) or (num_inter > num_cur*0.5):
            is_valid = 0
        
        return is_valid



    def get_video_track(self, frames, human_masks, mask_into_list, vis=False):

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor_sam2.init_state(frames)

            h, w, _ = frames[0].shape
            frames_mask = [np.zeros([h, w], dtype=np.uint8) for i in range(len(frames))]
            frames_mask_vis = [np.zeros_like(frames[0]) for i in range(len(frames))]
            frame_mask_static = None
            frame_vis_static = None

            # mask_into_list = mask_into_list[1:]
            for mask_info in mask_into_list:
                # add new prompts and instantly get the output on the same frame
                ann_frame_idx = mask_info['frame_idx']
                ann_obj_id = mask_info['obj_id']
                ann_mask = mask_info['mask']
                static = mask_info['static']
                # remove watercolor
                # ann_mask[-60:, -650:, :] = (0, 0, 0)

                is_valid = self.judge_obj_valid(ann_mask, frames_mask[ann_frame_idx])
                if is_valid == 0:
                    print('repeat obj, skip')
                    continue


                # mask = ann_mask[:, :, 0] > 128
                # mask = mask.astype(np.uint8)
                mask = ann_mask.astype(np.uint8) # h,w: 0,1
                # print('mask:', mask.shape)
                # print(np.max(mask))
                # print(np.min(mask))
                # print(mask.dtype)
                self.predictor_sam2.add_new_mask(
                    inference_state=state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=mask,
                )

                # propagate the prompts to get masklets throughout the video
                if static == 1:
                    max_frame_num_to_track = 10
                else:
                    max_frame_num_to_track = None

                frames_mask_tmp = [np.zeros([h, w], dtype=np.uint8) for i in range(len(frames))]

                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor_sam2.propagate_in_video(state,
                                                                                                max_frame_num_to_track=max_frame_num_to_track):
                    # print('out_frame_idx:', out_frame_idx)
                    # print('out_obj_ids:', out_obj_ids)
                    # print('out_mask_logits:', out_mask_logits.shape) # 1,1,h,w
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]  # h,w
                    # frames_mask.append((mask * 255).astype(np.uint8))
                    # frames_mask[out_frame_idx] = (mask * 255).astype(np.uint8)
                    frames_mask_tmp[out_frame_idx] = (mask * 255).astype(np.uint8)
                    frames_mask[out_frame_idx] = cv2.bitwise_or(frames_mask[out_frame_idx],
                                                                (mask * 255).astype(np.uint8))

                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor_sam2.propagate_in_video(state,
                                                                                                max_frame_num_to_track=max_frame_num_to_track,
                                                                                                reverse=True):
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]  # h,w
                    frames_mask_tmp[out_frame_idx] = (mask * 255).astype(np.uint8)
                    frames_mask[out_frame_idx] = cv2.bitwise_or(frames_mask[out_frame_idx],
                                                                (mask * 255).astype(np.uint8))

                if static == 1:
                    use_frame_idx = min(ann_frame_idx + 5, len(frames_mask) - 1)
                    if use_frame_idx == ann_frame_idx:
                        use_frame_idx = max(ann_frame_idx - 5, 0)

                    if frame_mask_static is None:
                        frame_mask_static = frames_mask_tmp[use_frame_idx]
                    else:
                        frame_mask_static = cv2.bitwise_or(frame_mask_static, frames_mask_tmp[use_frame_idx])

                # frames_mask[ann_frame_idx] = ann_mask[:,:,0]
                self.predictor_sam2.reset_state(state)

            # for idx in range(len(frames_mask)):
            #     mask = human_masks[idx]
            #     if len(mask.shape)==3:
            #         mask = mask[:,:,0]
            #     frames_mask[idx][mask>0] = 0

            if frame_mask_static is not None:
                for idx in range(len(frames_mask)):
                    frames_mask[idx] = cv2.bitwise_or(frames_mask[idx], frame_mask_static)

            if vis:
                for idx in range(len(frames_mask)):
                    mask = frames_mask[idx] / 255
                    frame = frames[idx]
                    painted_image = mask_painter(frame, mask.astype('uint8'), mask_alpha=0.8, mask_color=5)  # orange
                    frames_mask_vis[idx] = (painted_image).astype(np.uint8)

        return frames_mask, frames_mask_vis

    def mask_find_bboxs(self, mask):
        h, w = mask.shape[:2]
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,
                                                                            connectivity=8)  # connectivity参数的默认值为8
        stats = stats[stats[:, 4].argsort()]
        bboxs = stats[:-1]
        x_min, y_min, x_max, y_max = [], [], [], []
        for b in bboxs:
            x0, y0 = b[0], b[1]
            x1 = b[0] + b[2]
            y1 = b[1] + b[3]
            x_min.append(x0)
            y_min.append(y0)
            x_max.append(x1)
            y_max.append(y1)

        x0, y0 = min(x_min), min(y_min)
        x1, y1 = max(x_max), max(y_max)
        bbox_list = [max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)]
        return bbox_list

    def avg_depth_value(self, depth, mask):
        # depth hxw [0,255]  mask hxw [0,1]
        masked_depth = depth * mask
        # cv2.imwrite('./tmp/%s_mask_depth.png' % case, masked_depth)
        if np.sum(masked_depth > 0) == 0:
            return 0

        avg_depth_value = np.sum(masked_depth) / np.sum(masked_depth > 0)
        return avg_depth_value

    def get_obscure_prompts(self, mask, mask_all, depth, bbox, save_dir='output'):
        # mask hxw, [0,1]
        # depth hxw, [0,255]
        # bbox 1x4
        mask_list = []
        # mask_list.append(mask)

        avg_depth_value = self.avg_depth_value(depth, mask)
        new_depth = mask * avg_depth_value + (1 - mask) * depth
        # cv2.imwrite('./tmp_occ/%s_mask_depth_avg.png' % case, new_depth)

        mask_bbox = np.zeros_like(mask, dtype=np.uint8)
        mask_bbox[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        mask_obscure = np.where(new_depth > avg_depth_value, 1, 0)
        mask_obscure = (mask_obscure * mask_bbox).astype(np.uint8)
        cv2.imwrite('%s/mask_obscure.png'%save_dir, mask_obscure * 255.)

        sorted_masks = sorted(mask_all, key=(lambda x: x['area']), reverse=True)

        cc = 0
        count = 0
        for ann in sorted_masks:
            m = ann['segmentation'].astype(np.uint8)  # hxw, true false-> 1 0
            num_valid = np.sum(m > 0)
            intersect = cv2.bitwise_and(m, mask_obscure)
            # if np.sum(intersect) > 0:
            cv2.imwrite('%s/mask%d.png' % (save_dir, cc), m*255.)
            cc += 1

            # print("intersect ratio: {}, valid number:{}".format(np.sum(intersect), num_valid))
            if np.sum(intersect) > self.overlap_threshold * num_valid:
                mask_list.append(m)
                # count += 1

        return mask_list

    def get_obscure_obj(self, mask, mask_all, depth, bbox, sdc_mask, save_dir='output'):
        # mask hxw, [0,1]
        # depth hxw, [0,255]
        # bbox 1x4
        mask_list = []
        # mask_list.append(mask)

        mask = mask.astype(np.uint8) 

        avg_depth_value = self.avg_depth_value(depth, mask)
        new_depth = mask * avg_depth_value + (1 - mask) * depth
        # cv2.imwrite('./tmp_occ/%s_mask_depth_avg.png' % case, new_depth)
        
        mask_bbox = np.zeros_like(mask, dtype=np.uint8)
        mask_bbox[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        mask_obscure = np.where(new_depth > avg_depth_value, 1, 0)
        mask_obscure = (mask_obscure).astype(np.uint8)
        # cv2.imwrite('%s/mask_obscure.png'%save_dir, mask_obscure * 255.)
        # cv2.imwrite('%s/sdc_mask.png'%save_dir, sdc_mask*255.)

        sorted_masks = sorted(mask_all, key=(lambda x: x['area']), reverse=True)

        cc = 0
        count = 0
        num_sdc = np.sum(sdc_mask > 0)
        for ann in sorted_masks:
            m = ann['segmentation'].astype(np.uint8)  # hxw, true false-> 1 0

            is_valid = self.judge_obj_valid(mask, m)
            if is_valid==0:
                continue
            
            # ground layer or not
            IS_GROUND = 0
            ground_row = 10
            num_ground = np.sum(m[-ground_row:,:] > 0)
            if num_ground >= ground_row*m.shape[1]*0.9:
                IS_GROUND = 1

            m = clean_mask(m)
            cc = cc+1

            # cv2.imwrite('%s/mask%d_layer_seg.png' % (save_dir, cc), m*255.)
            
            inv_mask = (1-mask).astype(np.uint8)
            m = cv2.bitwise_and(m, inv_mask)

            
            # intersect_bbox = cv2.bitwise_and(m, mask_bbox)
            # if np.sum(intersect_bbox > 0) < 100:
            #     print('filter mask %d via bbox'%cc)
            #     continue

            num_valid = np.sum(m > 0)
            intersect = cv2.bitwise_and(m, mask_obscure)
            if np.sum(intersect) < self.overlap_threshold * num_valid:
                # print('filter mask %d via depth overlay'%cc)
                continue

            # cv2.imwrite('%s/mask%d_layer.png' % (save_dir, cc), m*255.)
            intersect_sdc = cv2.bitwise_and(intersect, sdc_mask)
            # cv2.imwrite('%s/mask%d_sdc.png' % (save_dir, cc), intersect_sdc*255.)
            # print('num_intersect', np.sum(intersect_sdc > 0))
            # print('therold:', int(num_sdc*0.1))

            if IS_GROUND==1:
                sdc_therold = int(num_sdc*0.1)
            elif np.sum(m > 0) > np.sum(mask > 0): # mask layer larger than mian object
                sdc_therold = 600
            else:
                sdc_therold = 100

            # if np.sum(intersect_sdc > 0) < int(num_sdc*0.05):
            if np.sum(intersect_sdc > 0) < sdc_therold:
            # num_intersect_sdc = np.sum(intersect_sdc > 0)
            # if  num_intersect_sdc < int(num_sdc*0.05) and num_intersect_sdc < int(num_valid*0.5):
                # print('filter mask %d via sdc overlay'%cc)
                continue
            mask_list.append(m)

            # num_intersect_sdc = np.sum(intersect_sdc > 0)
            # if num_intersect_sdc > int(num_sdc*0.05) or num_intersect_sdc > int(num_valid*0.5):
            #     mask_list.append(m)
            # else:
            #     print('filter mask %d via sdc overlay'%cc)
            #     continue

        return mask_list

    def detect_text_bbox(self, image_cv2, input_box, mask_person, thresh=0.2):
        bbox_list = []
        image_masked = np.zeros_like(image_cv2, dtype=np.uint8)
        img = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        mask_person = mask_person.astype(np.uint8)

        height, width, _ = image_cv2.shape
        image_resize = cv2.resize(image_cv2, (800, 800))
        image_resize = image_resize - np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image_resize /= 255.
        image_resize = np.expand_dims(image_resize.transpose(2, 0, 1), axis=0)

        outputs = self.dbnet_session.run(['pred'], {'images': image_resize})
        pred = outputs[0]
        segmentation = pred > thresh
        boxes, scores = boxes_from_bitmap(pred, segmentation, width,
                                          height, is_numpy=True)

        if len(boxes) == 0:
            return None, None
        else:
            for i in range(len(boxes)):
                bbox = boxes[i]
                new_bbox = [bbox[0], bbox[1], bbox[4], bbox[5]]
                mask_tmp = np.zeros_like(mask_person, dtype=np.uint8)
                mask_tmp[bbox[1]:bbox[5], bbox[0]:bbox[4]] = 1
                mask_union = cv2.bitwise_or(mask_tmp, mask_person)
                # add depth
                if is_rectangle_intersect(input_box, new_bbox) and np.sum(mask_union - mask_person) > 0:
                    bbox_list.append(new_bbox)
                    image_masked[bbox[1]:bbox[5], bbox[0]:bbox[4]] = img[bbox[1]:bbox[5], bbox[0]:bbox[4]]
        return bbox_list, image_masked

    def load_video(self, vid_path):
        # load video using imageio
        reader = imageio.get_reader(vid_path)
        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
        reader.close()
        return frames

    def convert2rgb(self, mask):
        if len(mask.shape) == 2:
            mask = mask.reshape(h, w, 1).repeat(3, axis=-1)  # np.stack([mask, mask, mask], -1)
        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask.repeat(3, axis=-1)

        return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment with depth')

    parser.add_argument('--vid_dir', type=str, default='./assets/test_motion/test_motion_0730_processed0805/vid')
    parser.add_argument('--save_dir', type=str, default='./tmp_occ')

    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--sam2_model_type', type=str, default='hiera_l',
                        choices=['hiera_l', 'hiera_s', 'hiera_t', 'hiera_b+'])
    parser.add_argument('--depth_model_type', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--sam2_model_cfg', type=str, default='sam2_hiera_l.yaml',
                        choices=['sam2_hiera_l.yaml', 'sam2_hiera_s.yaml', 'sam2_hiera_t.yaml', 'sam2_hiera_b+.yaml'])

    parser.add_argument('--detect_cfg_path', type=str, default='./hmr2/configs/cascade_mask_rcnn_vitdet_h_75ep.py')
    parser.add_argument('--detect_ckpt_path', type=str,
                        default='models/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl')
    parser.add_argument('--dbnet_ckpt_path', type=str, default='./models/dbnet.onnx')

    parser.add_argument('--overlap_threshold', type=float, default=0.4)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--debug', dest='debug', action='store_true', help='save debug images')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--detect_text', type=bool, default=False)

    args = parser.parse_args()

    # initialise models
    sam_model_type = args.sam_model_type
    sam2_model_cfg = args.sam2_model_cfg
    sam2_model_type = args.sam2_model_type
    depth_model_type = args.depth_model_type
    detect_cfg_path = args.detect_cfg_path
    detect_ckpt_path = args.detect_ckpt_path
    dbnet_ckpt_path = args.dbnet_ckpt_path
    overlap_threshold = args.overlap_threshold
    device = args.device
    detect_text = args.detect_text

    vid_dir = args.vid_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    debug = args.debug
    debug = True
    grayscale = args.grayscale

    segmenter = Segmenter()

    vid_path = 'output/vid.mp4'
    print('processing:', vid_path)
    case = os.path.basename(vid_path)[:-4]

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    frames = segmenter.load_video(vid_path)

    start_idx = 0
    frames = frames[start_idx:]

    print("number of frames:", len(frames))
    image = np.array(frames[0])  # rgb first frame
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, c = image.shape

    input_box = segmenter.detect_person(img_cv2)  # x1,y1,x2,y2
    print('input_box:', input_box)
    if input_box is None:
        print("no person detected")

    # set image for sam
    segmenter.set_image_sam(image)
    # segment person
    mask = segmenter.pred_sam(input_box)
    mask_rgb = mask.reshape(h, w, 1).repeat(3, axis=-1)
    mask_demo = (mask_rgb * 255.).astype(np.uint8)
    ## vis seg
    # painted_image = mask_painter(image, mask, background_alpha=0.8)
    painted_image = mask_painter(image, mask.astype('uint8'), mask_alpha=0.8, mask_color=5)  # orange
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite(os.path.join(save_dir, '%s.png' % case), painted_image)

    segmenter.reset_image_sam()

    # automask
    # mask_all = segmenter.pred_automask(image)
    mask_all = segmenter.pred_automask_large(image)
    print('mask_all:', mask_all[-1])
    mask_anno = show_automask_anno(mask_all, input_box, text_box_list=None)
    if debug:
        cv2.imwrite(os.path.join(save_dir, '%s_mask_anno.png' % case), mask_anno)
        print('saved to ', os.path.join(save_dir, '%s_mask_anno.png' % case))

    # depth image
    depth = segmenter.pred_depth(img_cv2)
    depth_demo = np.repeat(depth[..., np.newaxis], 3, axis=-1)  # h,w,3

    # demo image mask depth automask
    if debug:
        split_region = np.ones((img_cv2.shape[0], 20, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat(
            [img_cv2, mask_anno, split_region, split_region, mask_demo, split_region, depth_demo])
        cv2.imwrite(os.path.join(save_dir, '%s_demo.png' % case), combined_result)

    mask_list = segmenter.get_obscure_prompts(mask, mask_all, depth, input_box)
    print("mask numbers:", len(mask_list))  # 0 is person
    if debug:
        for i in range(len(mask_list)):
            cv2.imwrite(os.path.join(save_dir, '%s_mask%s.png' % (case, str(i))), mask_list[i] * 255.)

    print("start process video segment...")

    # video track the masked object
    mask_into_list = []
    for obj_idx, obj_mask in enumerate(mask_list):
        mask_info = {}
        frame_idx = 0
        mask_info['mask'] = obj_mask
        mask_info['frame_idx'] = frame_idx
        mask_info['obj_id'] = obj_idx
        mask_info['static'] = 0
        # print('mask_info:', mask_info)
        mask_into_list.append(mask_info)

    vis_mask = True
    frames_mask, frames_mask_vis = segmenter.get_video_track(frames, mask_into_list, vis=vis_mask)

    imageio.mimsave('tmp_occ/occ.mp4', frames_mask, fps=30, quality=8, macro_block_size=1)
    imageio.mimsave('tmp_occ/occ_vis.mp4', frames_mask_vis, fps=30, quality=8, macro_block_size=1)

    # seg_dir = os.path.join(save_dir, 'seg')
    # os.makedirs(seg_dir, exist_ok=True)
    # dep_dir = os.path.join(save_dir, 'dep')
    # os.makedirs(dep_dir, exist_ok=True)
    # occlude_dir = os.path.join(save_dir, 'occ')
    # os.makedirs(occlude_dir, exist_ok=True)
    #
    # frames_occ = segmenter.pred_sam2_depth(frames, mask_list, seg_dir, dep_dir, occlude_dir)
    #
    # print('frames:', len(frames))
    # print('frames_occ:', len(frames_occ))



    # frames_occ_new = []
    # for i in range(start_idx):
    #     frame_add = np.zeros_like(frames_occ[0])
    #     frames_occ_new.append(frame_add)

    # for frame in frames_occ:
    #     frames_occ_new.append(frame)

    # outpath = 'output/occ.mp4'
    # imageio.mimsave(outpath, frames_occ_new, fps=30, quality=8, macro_block_size=1)

    # print('saved to ', outpath)

    # reset sam2
    # segmenter.reset_video_sam2()







