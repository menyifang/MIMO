import torch
import cv2
import numpy as np
import os, imageio, glob
import tqdm
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tools.painter import mask_painter, show_automask_anno
from tools.util import *
from tools.transforms import mat2aa, aa2mat
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import MetadataCatalog
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'segment-anything-2-main'))
from sam2.build_sam import build_sam2_video_predictor

from vitpose_model import ViTPoseModel
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hamer.models import load_hamer
from hamer.datasets.vitdet_dataset import ViTDetDataset as ViTDetDataset_hamer
from ProPainter.infer import Propainter
import trimesh
import smplx
import json
from demo_occ import OccTracker
import time

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
kernel = np.ones((25, 25), dtype=np.uint8)

import onnxruntime

if_gpu = onnxruntime.get_device()
if if_gpu == 'GPU':
    providers = ['CUDAExecutionProvider']
else:
    providers = ['CPUExecutionProvider']
refine_session = onnxruntime.InferenceSession("models/refine_mask.onnx", providers=providers)
refine_input_name = refine_session.get_inputs()[0].name
refine_output_name = [output.name for output in refine_session.get_outputs()]


class DefaultPredictor_Lazy:
    """Create a simple end-to-end predictor with the given config that runs on single device for a
    single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).
    2. Always take BGR image as the input and apply format conversion internally.
    3. Apply resizing defined by the config (`cfg.INPUT.{MIN,MAX}_SIZE_TEST`).
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            test dataset name in the config.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: a yacs CfgNode or a omegaconf dict object.
        """
        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)  # noqa: F821
            if len(cfg.DATASETS.TEST):
                test_dataset = cfg.DATASETS.TEST[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

            self.input_format = cfg.INPUT.FORMAT
        else:  # new LazyConfig
            self.cfg = cfg
            self.model = instantiate(cfg.model)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        self.model.eval().cuda()
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class BaseSegmenter:
    def __init__(self, SAM_checkpoint, model_type, device='cuda:0'):
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h
        """
        print(f"Initializing BaseSegmenter to {device}")
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], 'model_type must be vit_b, vit_l, or vit_h'

        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model = sam_model_registry[model_type](checkpoint=SAM_checkpoint)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded = False

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.orignal_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return

    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.predictor.reset_image()
        self.embedded = False

    def predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both', 'bbox'], 'mode must be point, mask, or both'

        if mode == 'point':
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'],
                                                           point_labels=prompts['point_labels'],
                                                           multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.predictor.predict(mask_input=prompts['mask_input'],
                                                           multimask_output=multimask)
        elif mode == 'both':  # both
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'],
                                                           point_labels=prompts['point_labels'],
                                                           mask_input=prompts['mask_input'],
                                                           multimask_output=multimask)
        elif mode == 'bbox':  # both
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'],
                                                           point_labels=prompts['point_labels'],
                                                           # box=input_box[None, :],
                                                           box=prompts['box'],
                                                           multimask_output=multimask)
        else:
            raise ("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits


def save_results(dir, frames_src, frames_mask, frames_bbox, frames_mask_vis, frames_sdc, frames_bk, frames_occ,
                 frames_occ_vis):
    if frames_src is not None and len(frames_src) > 0:
        outpath = os.path.join(dir, 'vid.mp4')
        imageio.mimsave(outpath, frames_src, fps=30, quality=8, macro_block_size=1)

    # save bbox params
    if frames_bbox is not None and len(frames_bbox) > 0:
        outpath = os.path.join(dir, 'bbox.npy')
        np.save(outpath, frames_bbox)

    # save mask
    if frames_mask is not None and len(frames_mask) > 0:
        outpath = os.path.join(dir, 'mask.mp4')
        imageio.mimsave(outpath, frames_mask, fps=30, quality=8, macro_block_size=1)

    # save frames_mask_vis
    if frames_mask_vis is not None and len(frames_mask_vis) > 0:
        outpath = os.path.join(dir, 'mask_vis.mp4')
        imageio.mimsave(outpath, frames_mask_vis, fps=30, quality=8, macro_block_size=1)

    # save sdc
    if frames_sdc is not None and len(frames_sdc) > 0:
        outpath = os.path.join(dir, 'sdc.mp4')
        imageio.mimsave(outpath, frames_sdc, fps=30, quality=8, macro_block_size=1)

    # save bk
    if frames_bk is not None and len(frames_bk) > 0:
        outpath = os.path.join(dir, 'bk.mp4')
        imageio.mimsave(outpath, frames_bk, fps=30, quality=8, macro_block_size=1)

    if frames_occ is not None and len(frames_occ) > 0:
        outpath = os.path.join(dir, 'occ.mp4')
        imageio.mimsave(outpath, frames_occ, fps=30, quality=8, macro_block_size=1)

    if frames_occ_vis is not None and len(frames_occ_vis) > 0:
        outpath = os.path.join(dir, 'occ_vis.mp4')
        imageio.mimsave(outpath, frames_occ_vis, fps=30, quality=8, macro_block_size=1)

    print('saved human tracking results to:', dir)


def refine_img_prepross(image, mask):
    im_ary = np.asarray(image).astype(np.float32)
    input = np.concatenate([im_ary, mask[:, :, np.newaxis]], axis=-1)
    return input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MAX_FRAME_NUM", type=int, default=150)
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--fp16', action='store_true',
        help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')
    args = parser.parse_args()
    return args


class VideoProcessor():
    def __init__(self, model_path='models', debug_mode=False):
        args = parse_args()
        self.args = args

        from detectron2.config import LazyConfig  # install detectron2 first
        cfg_path = Path(
            'hmr2/configs/cascade_mask_rcnn_vitdet_h_75ep.py')
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = os.path.join(model_path,
                                                            'detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl')
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # Load BaseSegmenter
        SAM_checkpoint = os.path.join(model_path, 'sam_vit_h_4b8939.pth')
        model_type = 'vit_h'
        device = "cuda:0"
        self.base_segmenter = BaseSegmenter(SAM_checkpoint=SAM_checkpoint, model_type=model_type, device=device)

        ## Load video_tracker
        checkpoint = os.path.join(model_path, 'sam2_hiera_large.pt')
        model_cfg = "sam2_hiera_l.yaml"
        self.sam2_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        # Load HMR2
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_hmr2, self.model_cfg = load_hmr2(
            os.path.join(model_path, 'hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'))
        self.model_hmr2 = self.model_hmr2.to(self.device)
        self.model_hmr2.eval()

        # Load Hamer
        self.model_hamer, self.model_cfg_hamer = load_hamer(os.path.join(model_path, 'hamer/checkpoints/hamer.ckpt'))
        self.model_hamer = self.model_hamer.to(self.device)
        self.model_hamer.eval()

        # Load pose estimator
        self.cpm = ViTPoseModel(self.device)

        # Load smpl related
        vc_info = np.load('./assets/sdc_info.npy', allow_pickle=True)
        self.vc_smpl = vc_info.item().get('smpl')

        part_segm_filepath = './assets/smpl_vert_segmentation.json'
        part_segm = json.load(open(part_segm_filepath))
        left_hand_idxs = []
        right_hand_idxs = []
        hand_idxs = []
        for key in ['leftHand', 'leftHandIndex1']:
            left_hand_idxs += part_segm[key]
        for key in ['rightHand', 'rightHandIndex1']:
            right_hand_idxs += part_segm[key]
        for key in ['leftHand', 'leftHandIndex1', 'rightHand', 'rightHandIndex1']:
            hand_idxs += part_segm[key]
        self.left_hand_idxs = left_hand_idxs
        self.right_hand_idxs = right_hand_idxs
        self.hand_idxs = hand_idxs

        smplh_model_path = './assets'
        body_model = dict(
            model_path=smplh_model_path,
            model_type='smplh',
            gender='neutral',
            ext='npz',
            use_pca=False,
            flat_hand_mean=True)
        self.smpl_model = smplx.create(**body_model)

        # Load render
        smpl_path = 'assets/smpl/01.ply'
        smpl_vertices, smpl_faces = load_ply(smpl_path)
        self.renderer = Renderer(cfg=None, faces=smpl_faces)

        # Load scene recovery
        self.args.fp16 = True
        self.model_bk = Propainter(self.args, model_path=os.path.join(model_path, 'propainter_weights'))

        # Load occ extraction
        self.model_occ = OccTracker()

    def get_first_mask(self, frames):
        CODE = 0  # 0,success; 1, no person; 2, min person; 3, no full-body person
        mask_seg = None
        NO_PERSON = 0
        MIN_PERSON = 0
        HALF_PERSON = 0
        for frame_idx, frame in enumerate(frames):
            if mask_seg is not None:
                break
            img_cv2 = frame[:, :, ::-1]
            # Detect humans in image
            det_out = self.detector(img_cv2)
            det_instances = det_out['instances']
            # print('det_instances.scores:', det_instances.scores)
            valid_idx = (det_instances.pred_classes == 0)
            valid_idx = valid_idx & (det_instances.scores > 0.95)  # to be adjust
            # print('valid_idx:', valid_idx)
            if valid_idx.sum() == 0:
                print('skip, no person detected')
                NO_PERSON = 1
                continue

            valid_scores = det_instances.scores[valid_idx].cpu().numpy()
            valid_boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()  # 2，4

            # np.save('human_boxes.npy', boxes)
            # print('img_cv2:', img_cv2.shape)

            # filter out the boxes with low area
            areas = (valid_boxes[:, 2] - valid_boxes[:, 0]) * (valid_boxes[:, 3] - valid_boxes[:, 1])
            valid_idx = (areas > 0.02 * img_cv2.shape[0] * img_cv2.shape[1])
            valid_boxes = valid_boxes[valid_idx]
            valid_scores = valid_scores[valid_idx]

            if len(valid_boxes) == 0:
                print('skip, no person large enough detected')
                MIN_PERSON = 1
                continue

            sorted_idx = np.argsort(areas[valid_idx])[::-1]
            valid_boxes = valid_boxes[sorted_idx]
            valid_scores = valid_scores[sorted_idx]

            # filter out invalid person without full-body
            vitposes_out = self.cpm.predict_pose(
                frame,
                [np.concatenate([valid_boxes, valid_scores[:, None]], axis=1)],
            )
            kps = []
            for vitposes in vitposes_out:
                kps.append(vitposes['keypoints'])
            kps = np.array(kps)  # n,133,3
            # print('kps:', kps.shape)
            # print(kps[:, :17, :])
            valid_person_ind = get_valid_person(kps, threshold=0.35)
            # print('valid_person_ind:', valid_person_ind)
            valid_person_num = len(valid_person_ind)
            if valid_person_num < 1:
                print('skip, no full-body person detected')
                HALF_PERSON = 1
                continue

            # boxes = boxes[valid_person_ind]
            main_id = valid_person_ind[0]
            print('main_id:', main_id)

            # # # get the main person
            # if len(boxes) > 1:
            #     main_id = get_main_person(boxes, img_cv2)
            #     print('main_id:', main_id)
            # else:
            #     main_id = 0
            # # main_id = 0

            person_bbox = valid_boxes[main_id]

            ### segment the human
            self.base_segmenter.set_image(frame)
            mode = 'bbox'
            input_box = np.array(person_bbox)  # w1,h1,w2,h2
            prompts = {
                'point_coords': None,
                'point_labels': None,
                'box': input_box[None, :],
                'multimask_output': False,
            }
            masks, scores, logits = self.base_segmenter.predict(prompts, mode,
                                                                multimask=False)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
            self.base_segmenter.reset_image()
            mask_track = masks[np.argmax(scores)].astype('uint8')
            mask_seg = mask_track.copy()  # 0,1 mask

        print('start frame_idx:', frame_idx - 1)
        frames = frames[frame_idx - 1:]

        if mask_seg is None:
            if MIN_PERSON == 1:
                CODE = 2
            elif HALF_PERSON == 1:
                CODE = 3
            else:
                CODE = 1

        return frames, mask_seg, CODE

    def get_video_track(self, frames, mask, vis=False):

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.sam2_predictor.init_state(frames)

            # add new prompts and instantly get the output on the same frame
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            points, labels = sample_points_mask_unified(mask)

            # # vis points
            # vis_points = draw_points(mask, points, labels)
            # cv2.imwrite('vis_sample_points.png', vis_points)

            frame_idx, object_ids, masks = self.sam2_predictor.add_new_points(state,
                                                                              frame_idx=ann_frame_idx,
                                                                              obj_id=ann_obj_id,
                                                                              points=points,
                                                                              labels=labels, )

            # # vis the mask of first frame
            # print('masks:', masks.shape)
            # tmp_mask = (masks[0] > 0.0).cpu().numpy()[0] # h,w
            # cv2.imwrite('res_mask.png', (tmp_mask*255).astype(np.uint8))

            # propagate the prompts to get masklets throughout the video
            video_segments = {}  # video_segments contains the per-frame segmentation results
            frames_mask = []
            frames_mask_vis = []
            frames_final = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(state):
                # print('out_frame_idx:', out_frame_idx)
                # print('out_obj_ids:', out_obj_ids)
                # print('out_mask_logits:', out_mask_logits.shape) # 1,1,h,w
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]  # h,w
                frames_mask.append((mask * 255).astype(np.uint8))

                frame = frames[out_frame_idx]
                frames_final.append(frame)

                if vis:
                    painted_image = mask_painter(frame, mask.astype('uint8'), mask_alpha=0.8, mask_color=5)  # orange
                    # painted_image = mask_painter(frame, mask.astype('uint8'), mask_alpha=0.8, mask_color=3) # blue
                    frames_mask_vis.append((painted_image).astype(np.uint8))

            self.sam2_predictor.reset_state(state)

        return frames_final, frames_mask, frames_mask_vis

    def get_human(self, frames):

        # get mask of the first frame
        frames, fisrt_mask, CODE = self.get_first_mask(frames)
        if CODE != 0:
            return None, None, None, CODE
        # truncate the frames
        n_frame = len(frames)
        if n_frame > self.args.MAX_FRAME_NUM:
            frames = frames[:self.args.MAX_FRAME_NUM]

        # track the masked human in the video
        vis_mask = True
        frames_src, frames_mask, frames_mask_vis = self.get_video_track(frames, fisrt_mask, vis=vis_mask)
        return frames_src, frames_mask, frames_mask_vis, 0

    def get_bbox(self, frames_src, frames_mask):
        # get bbox from mask
        frames_bbox = []
        for frame, mask in zip(frames_src, frames_mask):
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = clean_mask(mask)
            bbox_xyxy = get_bbox_from_mask(mask, frame, pad_h=0.01, pad_w=0.01)  # 1,4
            pred_scores = np.ones((1, 1))  # 1,1
            bbox_info = {}
            bbox_info['bbox'] = bbox_xyxy
            bbox_info['score'] = pred_scores
            frames_bbox.append(bbox_info)
        return frames_bbox

    def remove_part_verts(self, smpl_vertices, smpl_faces, hand_idxs):
        smpl_mesh = trimesh.Trimesh(smpl_vertices, smpl_faces, vertex_colors=self.vc_smpl, process=False,
                                    maintains_order=True)
        hand_mask = torch.zeros(smpl_vertices.shape[0], )
        hand_mask.index_fill_(0, torch.tensor(hand_idxs), 1.0)
        hand_mask = 1 - hand_mask
        hand_mesh = apply_vertex_mask(smpl_mesh.copy(), hand_mask)
        return hand_mesh

    def get_motion(self, frames, frames_bbox):
        frames_sdc = []
        frame_idx = 0

        for frame in tqdm.tqdm(frames):
            img_cv2 = frame[:, :, ::-1]
            img = img_cv2.copy()[:, :, ::-1]

            bbox_xyxy = frames_bbox[frame_idx]['bbox']
            pred_scores = frames_bbox[frame_idx]['score']
            if (bbox_xyxy.sum() == 0):
                render_res = np.zeros_like(frame)
                frames_sdc.append(render_res)

                frame_idx += 1
                continue

            # Detect human keypoints for each person
            # cv2.imwrite(os.path.join(save_dir, f'{frame_idx}_img.png'), img)
            vitposes_out = self.cpm.predict_pose(
                img,
                [np.concatenate([bbox_xyxy, pred_scores], axis=1)],
            )
            # print('vitposes_out:', vitposes_out)

            bboxes = []
            is_right = []
            is_right_full = []

            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # # vis keypoint
                # frame_keypoint = draw_keypoints(frame_bbox, left_hand_keyp)
                # cv2.imwrite(os.path.join(save_dir, f'{frame_idx}_bbox_kp.png'), frame_keypoint)

                # Rejecting not confident detections, save as left, right, ..., left, right
                keyp = left_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
                    is_right_full.append(0)
                else:
                    is_right_full.append(100)
                keyp = right_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)
                    is_right_full.append(1)
                else:
                    is_right_full.append(100)

            if len(bboxes) == 0:
                print('No tracking person hand in frame:', frame_idx)
                # break

            if len(bboxes) != 0:
                boxes = np.stack(bboxes)
                right = np.stack(is_right)
            else:
                boxes = None
                right = None

            # Run HMR2.0 on all detected humans
            dataset = ViTDetDataset(self.model_cfg, img_cv2, bbox_xyxy)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            # only track and process one person
            for batch in dataloader:
                # print('batch:', batch)
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self.model_hmr2(batch)

                pred_cam = out['pred_cam']
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                                   scaled_focal_length).detach().cpu().numpy()

                # visualize body model parameters
                pred_smpl_params = out['pred_smpl_params']

                # Render the result
                batch_size = batch['img'].shape[0]
                # print('batch_size:', batch_size)
                for n in range(batch_size):
                    # Get filename from path img_path
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                            DEFAULT_STD[:, None, None] / 255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                            DEFAULT_MEAN[:, None, None] / 255)
                    input_patch = input_patch.permute(1, 2, 0).numpy()

                    # Add all verts and cams to list
                    # verts = out['pred_vertices'][n].detach().cpu().numpy()

                    cam_t = pred_cam_t_full[n]

            # save params
            info_dict = {}
            info_dict['params'] = pred_smpl_params
            info_dict['cam_t'] = cam_t
            info_dict['img_size'] = img_size[0]
            info_dict['focal_length'] = scaled_focal_length
            # info_path = os.path.join(case_dir, '%04d.npy'%frame_idx)
            # np.save(info_path, info_dict)
            # frames_body.append(info_dict)

            ### process hand
            if boxes is None:
                pred_mano_params = None
            else:
                # Run reconstruction on all detected hands
                dataset_hamer = ViTDetDataset_hamer(self.model_cfg_hamer, img_cv2, boxes, right, rescale_factor=2.0)
                dataloader_hamer = torch.utils.data.DataLoader(dataset_hamer, batch_size=8, shuffle=False,
                                                               num_workers=0)
                for batch in dataloader_hamer:
                    # print('batch:', batch['personid'])
                    batch = recursive_to(batch, self.device)
                    with torch.no_grad():
                        out = self.model_hamer(batch)

                    multiplier = (2 * batch['right'] - 1)
                    pred_cam = out['pred_cam']
                    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    multiplier = (2 * batch['right'] - 1)
                    scaled_focal_length = self.model_cfg_hamer.EXTRA.FOCAL_LENGTH / self.model_cfg_hamer.MODEL.IMAGE_SIZE * img_size.max()
                    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                                       scaled_focal_length).detach().cpu().numpy()
                    # print hand params
                    pred_mano_params = out['pred_mano_params']

            # save params
            hand_info_dict = {}
            hand_info_dict['params'] = pred_mano_params
            hand_info_dict['right'] = is_right_full
            # frames_hand.append(hand_info_dict)

            ### render sdc
            smpl_param = info_dict['params']
            all_cam_t = info_dict['cam_t']
            img_size = info_dict['img_size']
            scaled_focal_length = info_dict['focal_length']
            body_pose = smpl_param['body_pose']
            bs = body_pose.shape[0]
            body_pose_aa = mat2aa(body_pose)[:, :21, :].reshape(bs, -1).to('cpu')
            global_orient = mat2aa(smpl_param['global_orient']).reshape(bs, -1).to('cpu')
            betas = smpl_param['betas'].reshape(bs, -1).to('cpu')

            # load hand info
            mano_param = hand_info_dict['params']
            all_right = hand_info_dict['right']  # len:3

            # compute hand pose
            right_hand_pose = torch.zeros([bs, 15 * 3], dtype=torch.float32)
            left_hand_pose = torch.zeros([bs, 15 * 3], dtype=torch.float32)
            right_hand_global_orient = body_pose[:, -1, :, :]  # bs,3,3
            left_hand_global_orient = body_pose[:, -2, :, :]  # bs,3,3
            right_hand_valid = torch.zeros([bs], dtype=torch.uint8)
            left_hand_valid = torch.zeros([bs], dtype=torch.uint8)

            hand_idx = 0
            fetch_idx = 0
            for is_right in all_right:
                person_idx = hand_idx // 2
                if is_right > 1:
                    hand_idx += 1
                    continue
                else:
                    hand_pose = mat2aa(mano_param['hand_pose'][fetch_idx]).reshape(1, -1).to('cpu')
                    hand_global_orient = (mano_param['global_orient'][fetch_idx])

                    if is_right < 1:  # left hand
                        hand_pose[:, 1::3] *= -1
                        hand_pose[:, 2::3] *= -1
                        left_hand_pose[person_idx] = hand_pose

                        hand_global_orient = mat2aa(hand_global_orient)
                        hand_global_orient[:, 1] *= -1
                        hand_global_orient[:, 2] *= -1
                        left_hand_global_orient[person_idx] = aa2mat(hand_global_orient)
                        left_hand_valid[person_idx] = 1

                    else:  # right hand
                        right_hand_pose[person_idx] = hand_pose
                        right_hand_global_orient[person_idx] = hand_global_orient
                        right_hand_valid[person_idx] = 1

                left_hand_valid = left_hand_valid > 0
                right_hand_valid = right_hand_valid > 0

                hand_idx += 1
                fetch_idx += 1

            right_tree_idx = [2, 5, 8, 13, 16, 18]
            tree_rot_mats = body_pose[:, right_tree_idx]
            right_wrist_local = compute_wrist_local_pose(smpl_param['global_orient'][:, 0, :, :], tree_rot_mats,
                                                         right_hand_global_orient)
            right_wrist_local = mat2aa(right_wrist_local).reshape(bs, -1).to('cpu')

            if right_hand_valid is not None:
                body_pose_aa[right_hand_valid, -3:] = right_wrist_local[right_hand_valid]

            left_tree_idx = [2, 5, 8, 12, 15, 17]
            tree_rot_mats = body_pose[:, left_tree_idx]
            left_wrist_local = compute_wrist_local_pose(smpl_param['global_orient'][:, 0, :, :], tree_rot_mats,
                                                        left_hand_global_orient)
            left_wrist_local = mat2aa(left_wrist_local).reshape(bs, -1).to('cpu')
            # body_pose_aa[:, -6:-3] = left_wrist_local
            if left_hand_valid is not None:
                body_pose_aa[left_hand_valid, -6:-3] = left_wrist_local[left_hand_valid]

            output = self.smpl_model(  # expression=expression,
                betas=betas, global_orient=global_orient,
                body_pose=body_pose_aa,
                left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                return_verts=True)
            # print(output.vertices.shape)

            ### render sdc image
            all_verts = []
            all_faces = []
            all_colors = []
            for i in range(output.vertices.shape[0]):
                verts = output.vertices[i].detach().cpu().numpy()
                faces = self.smpl_model.faces
                colors = self.vc_smpl.copy()
                if (left_hand_valid[i] == False and right_hand_valid[i] == False):
                    # print('remove left and right hand')
                    mesh_removed = self.remove_part_verts(verts, faces, self.hand_idxs)
                    verts = mesh_removed.vertices
                    faces = mesh_removed.faces
                    colors = mesh_removed.visual.vertex_colors
                elif (left_hand_valid[i] == False):
                    # print('remove left hand')
                    mesh_removed = self.remove_part_verts(verts, faces, self.left_hand_idxs)
                    verts = mesh_removed.vertices
                    faces = mesh_removed.faces
                    colors = mesh_removed.visual.vertex_colors
                elif (right_hand_valid[i] == False):
                    # print('remove right hand')
                    mesh_removed = self.remove_part_verts(verts, faces, self.right_hand_idxs)
                    verts = mesh_removed.vertices
                    faces = mesh_removed.faces
                    colors = mesh_removed.visual.vertex_colors

                all_verts.append(verts)
                all_faces.append(faces)
                all_colors.append(colors)

            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(0, 0, 0),
                focal_length=scaled_focal_length,
            )
            cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=[all_cam_t], render_res=img_size,
                                                          all_faces=all_faces, all_colors=all_colors,
                                                          **misc_args)

            render_res = (cam_view[:, :, :3] * 255).astype(np.uint8)
            # render_res = render_res * 0.5 + frame * 0.5
            # cv2.imwrite('vis_render_%04d.png' % frame_idx, render_res[:, :, ::-1])
            frames_sdc.append(render_res)

            frame_idx += 1

        # delete cache
        # del frames_body, frames_hand
        # del deepsort
        # del cpm
        return frames_sdc

    def get_bk_recover(self, frames, frames_mask, frames_bbox):
        args = self.args
        bbox_clip = get_clip_bbox(frames_bbox)
        frames_crop, frames_mask_crop, bbox_clip_final = crop_human(frames, frames_mask, bbox_clip)

        # if_single = check_single_human(frames_res_mask, bbox_list)
        w, h = frames_crop[0].size
        MAX_SIZE = 480
        resize_ratio_init = MAX_SIZE / min(h, w)
        print('resize_ratio_init:', resize_ratio_init)
        args.resize_ratio = resize_ratio_init
        while True:
            try:
                frames_crop_inpaint = self.model_bk.process(frames_crop, frames_mask_crop, args)
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                args.resize_ratio = args.resize_ratio * 0.75
                print('retry resize_ratio %f' % args.resize_ratio)

        # frames_crop_inpaint = process(args, device, frames_crop, frames_mask_crop, fps, size, video_name)
        print('size after inpaint:', frames_crop_inpaint[0].shape)
        frames_res_bk = []
        for img_idx, frame in enumerate(frames):
            frame_ori = np.array(frame)
            frames_crop = np.array(frames_crop_inpaint[img_idx])
            x1, y1, x2, y2 = bbox_clip_final
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h = y2 - y1
            w = x2 - x1
            frames_crop = cv2.resize(frames_crop, (w, h), interpolation=cv2.INTER_AREA)
            frame_ori[y1:y2, x1:x2] = frames_crop
            frames_res_bk.append(frame_ori)

        return frames_res_bk

    def get_occ(self, frames, frames_mask, frames_bbox, frames_sdc, save_dir='output'):

        # sample_rate = 30
        # idxs = list(range(0, len(frames), sample_rate))
        # idxs = idxs+[len(frames)-1]
        idxs = get_occ_frame(frames_mask, num_frame=5, interval=20)

        print('idxs:', idxs)

        # idxs = [0]
        mask_into_list = []
        obj_idx = 0
        for idx in idxs:
            # print('Processing frame %d'%idx)
            image = frames[idx]
            sdc = frames_sdc[idx]
            input_box = np.array(frames_bbox[idx]['bbox'][0]).astype('int')
            mask = frames_mask[idx]
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            if np.max(mask) > 1:
                mask = mask / 255

            # input_box = [0, 0, image.shape[1], image.shape[0]]
            # mask_all = self.model_occ.pred_automask(image)
            mask_all = self.model_occ.pred_automask_large(image)
            mask_anno = show_automask_anno(mask_all, input_box, text_box_list=None)
            # cv2.imwrite('%s/%d_mask_anno.png'%(save_dir,idx), mask_anno)

            img_cv2 = image[..., ::-1]
            # depth image
            depth = self.model_occ.pred_depth(img_cv2)
            depth_demo = np.repeat(depth[..., np.newaxis], 3, axis=-1)  # h,w,3

            sdc_mask = clean_mask(extract_mask_sdc(sdc))
            mask_list = self.model_occ.get_obscure_obj(mask, mask_all, depth, input_box, sdc_mask, save_dir)
            # print("mask numbers:", len(mask_list))
            for i in range(len(mask_list)):
                # cv2.imwrite('%s/%d_mask%d.png' % (save_dir, idx, i), mask_list[i] * 255.)
                mask_info = {}
                mask_info['mask'] = mask_list[i]
                mask_info['frame_idx'] = idx
                mask_info['obj_id'] = obj_idx
                mask_info['static'] = 0
                mask_into_list.append(mask_info)
                obj_idx = obj_idx + 1

        print('mask_into_list:', (mask_into_list))
        if len(mask_into_list) == 0:
            return None, None

        # video track the masked object
        vis_mask = True
        frames_occ, frames_occ_vis = self.model_occ.get_video_track(frames, frames_mask, mask_into_list, vis=vis_mask)

        return frames_occ, frames_occ_vis

    def run(self, vid_path, save_dir):

        start_time_tot = time.time()

        vid_path_new = os.path.join(save_dir, 'vid.mp4')
        mask_path = os.path.join(save_dir, 'mask.mp4')
        mask_vis_path = os.path.join(save_dir, 'mask_vis.mp4')
        bbox_path = os.path.join(save_dir, 'bbox.npy')
        sdc_path = os.path.join(save_dir, 'sdc.mp4')
        bk_path = os.path.join(save_dir, 'bk.mp4')
        occ_path = os.path.join(save_dir, 'occ_ori.mp4')
        occ_vis_path = os.path.join(save_dir, 'occ_vis.mp4')
        occ_refine_path = os.path.join(save_dir, 'occ.mp4')
        conf_file_outpath = os.path.join(save_dir, 'config.json')

        frames = load_video_fixed_fps(vid_path, target_fps=30)
        # resize to 720P
        resolution = 720
        in_H, in_W, _ = frames[0].shape
        if in_H > resolution and in_W > resolution:
            for i in range(len(frames)):
                frames[i] = resize_image(frames[i], resolution)

        # get human from video
        print('[Start to track human...]')
        # self.get_human(frames)
        frames_src, frames_mask, frames_mask_vis, CODE = self.get_human(frames)
        torch.cuda.empty_cache()
        if CODE == 1:
            return 1, 'failed to detect the person'
        elif CODE == 2:
            return 1, 'failed to detect valid person, maybe person is too small'
        elif CODE == 3:
            return 1, 'Invalid video, failed to detect valid person, full-body motion is required'
        imageio.mimsave(vid_path_new, frames_src, fps=30, quality=8, macro_block_size=1)
        imageio.mimsave(mask_path, frames_mask, fps=30, quality=10, macro_block_size=1)
        imageio.mimsave(mask_vis_path, frames_mask_vis, fps=30, quality=8, macro_block_size=1)
        # get bbox from mask
        frames_bbox = self.get_bbox(frames_src, frames_mask)
        np.save(bbox_path, frames_bbox)

        # get motion rep. for human
        print('[Start to extract motion rep...]')
        frames_sdc = self.get_motion(frames_src, frames_bbox)
        torch.cuda.empty_cache()
        imageio.mimsave(sdc_path, frames_sdc, fps=30, quality=8, macro_block_size=1)
        if not os.path.exists(sdc_path):
            return 2, 'Internal Algorithm error, failed to extract motion rep.'

        # get recovered scene from video
        print('[Start to recover scene...]')
        frames_bk = self.get_bk_recover(frames_src, frames_mask, frames_bbox)
        torch.cuda.empty_cache()
        imageio.mimsave(bk_path, frames_bk, fps=30, quality=8, macro_block_size=1)
        if not os.path.exists(bk_path):
            return 2, 'Internal Algorithm error, failed to recover bk'

        # frames_src = load_video_fixed_fps(vid_path_new, target_fps=30)
        # frames_mask = load_video_fixed_fps(mask_path, target_fps=30)
        # frames_bbox = np.load(bbox_path, allow_pickle=True)
        # frames_sdc = load_video_fixed_fps(sdc_path, target_fps=30)

        # reload frames_mask from local
        # frames_mask = load_video_fixed_fps(mask_path, target_fps=30)

        # get occulusion from video
        print('[Start to extract occulusion...]')
        frames_occ, frames_occ_vis = self.get_occ(frames_src, frames_mask, frames_bbox, frames_sdc, save_dir)
        torch.cuda.empty_cache()
        if frames_occ is not None and frames_occ_vis is not None:
            imageio.mimsave(occ_path, frames_occ, fps=30, quality=10, macro_block_size=1)
            imageio.mimsave(occ_vis_path, frames_occ_vis, fps=30, quality=8, macro_block_size=1)

        # frames_occ = load_video_fixed_fps(occ_path, target_fps=30)

        # occ refinememt
        if frames_occ is not None:
            print('[Start to refine occ mask...]')
            start_time = time.time()
            frames_occ_refine = []
            for frame_idx, occ_mask in tqdm.tqdm(enumerate(frames_occ)):
                if len(occ_mask.shape) == 3:
                    occ_mask = occ_mask[:, :, 0]
                frame = frames_src[frame_idx]
                occ_refine = refine_session.run(refine_output_name,
                                                {refine_input_name: refine_img_prepross(frame, occ_mask)})
                occ_refine = occ_refine[0].astype(np.uint8)  # [0,255]
                frames_occ_refine.append(occ_refine)
            imageio.mimsave(occ_refine_path, frames_occ_refine, fps=30, quality=10, macro_block_size=1)
            print("occ refine time:{}".format(time.time() - start_time))

        # update config file
        conf_file_path = 'assets/config.json'
        with open(conf_file_path, 'r') as f:
            conf = json.load(f)
        conf["time_crop"]["start_idx"] = 0
        conf["time_crop"]["end_idx"] = len(frames_src)
        with open(conf_file_outpath, 'w') as f:
            json.dump(conf, f, indent=4)

        print("tot process time:{}".format(time.time() - start_time_tot))
        return 0, 'Success'


if __name__ == "__main__":
    model = VideoProcessor()

    vid_path = 'assets/test_vid.mp4'
    print('Processing video %s' % vid_path)

    template_name = os.path.basename(vid_path)[:-4]

    save_dir = os.path.join('output', template_name)
    os.makedirs(save_dir, exist_ok=True)

    status_code, message = model.run(vid_path, save_dir)  # 0: success; 1: invalid video; 2: algo error

    print('status_code:', status_code)
    print('message:', message)

