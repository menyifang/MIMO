
import imageio
import cv2
import numpy as np
import os
import torch
from sam2.build_sam import build_sam2_video_predictor
from painter import mask_painter

def load_video_fixed_fps(vid_path, target_fps=30):
    # Load video and get metadata
    reader = imageio.get_reader(vid_path)
    fps = round(reader.get_meta_data()['fps'])

    # Calculate the ratio of original fps to target fps to determine which frames to keep
    keep_ratio = fps / target_fps
    n_frames = reader.count_frames()
    keep_frames_indices = np.arange(0, n_frames, keep_ratio).astype(int)

    # Extract frames at the target frame rate
    frames = [reader.get_data(i) for i in keep_frames_indices if i < len(reader)]

    reader.close()
    return frames

def get_bbox_from_mask(mask):
    # find the bounding box
    x, y, w, h = cv2.boundingRect(mask)  # 91 85 554 1836
    y_max = y + h
    x_max = x + w
    y = max(0, y)
    y_max = min(mask.shape[0], y_max)
    x = max(0, x)
    x_max = min(mask.shape[1], x_max)
    return x,y,x_max,y_max

def sample_points_mask(mask):
    '''
    Sample points on the mask with valid pixels
    Args:
        mask:

    Returns:
        sample_points: np.array([[x1, y1],[x2,y2],...], dtype=np.float32)

    '''
    x1, y1, x2, y2 = get_bbox_from_mask(mask)
    ## unified sample points on bbox of mask with valid pixels, as positive samples
    n_points = 4
    x = np.linspace(x1, x2-1, n_points)
    y = np.linspace(y1, y2-1, n_points)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    # valid points
    valid_idx = mask[yy.astype(int), xx.astype(int)] > 0
    xx = xx[valid_idx]
    yy = yy[valid_idx]
    sample_points = []
    labels = []
    for i in range(len(xx)):
        sample_points.append((xx[i], yy[i]))
        labels.append(1)

    # # sample negative points
    # n_points = 10
    # x = np.linspace(0, mask.shape[1]-1, n_points)
    # y = np.linspace(0, mask.shape[0]-1, n_points)
    # xx, yy = np.meshgrid(x, y)
    # xx = xx.flatten()
    # yy = yy.flatten()
    # # valid points
    # valid_idx = mask[yy.astype(int), xx.astype(int)] == 0
    # xx = xx[valid_idx]
    # yy = yy[valid_idx]
    # for i in range(len(xx)):
    #     sample_points.append((xx[i], yy[i]))
    #     labels.append(0)

    # convert to np.array
    sample_points = np.array(sample_points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return sample_points, labels


def sample_points_mask_unified(mask):
    '''
    Sample points on the mask with valid pixels
    Args:
        mask:

    Returns:
        sample_points: np.array([[x1, y1],[x2,y2],...], dtype=np.float32)

    '''
    # get points'coordinate with valid pixels
    points = np.argwhere(mask > 0) # h_idx,w_idx
    # unified sample n points
    sample_points = []
    labels = []
    # uniform sample n_points from points
    n_points = 10
    idxs = np.linspace(0, points.shape[0]-1, n_points).astype(int)
    for idx in idxs:
        sample_points.append((points[idx][1], points[idx][0]))
        labels.append(1)
    # convert to np.array
    sample_points = np.array(sample_points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return sample_points, labels


def draw_points(mask, points, labels):
    # draw points with labels
    if len(mask.shape)==2:
        mask = mask[:, :, None].repeat(3, axis=2)
    
    for i in range(len(points)):
        x, y = points[i]
        label = labels[i]
        if label == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # mask to color image
        cv2.circle(mask, (int(x), int(y)), 3, color, -1)
    
    return mask

if __name__ == '__main__':
    vid_path = '/mnt/workspace/myf272609/qingyao/mycode/Moore-AnimateAnyone-0130/assets/test_motion_0730/LdDzrtRKUqQ&list=LL&index=9.mp4'

    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    frames = load_video_fixed_fps(vid_path)

    # frames, fisrt_mask = get_first_mask(frames, detector)

    first_frame = frames[0]
    # cv2.imwrite('vis_first_frame.png', first_frame[...,::-1])


    # mask_path = '../vis_first_mask.png'
    mask_path = 'vis_first_frame_mask.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        state = predictor.init_state(frames)

        # add new prompts and instantly get the output on the same frame
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # points = np.array([[210, 350]], dtype=np.float32), (w,h)
        # labels = np.array([1], np.int32)# for labels, `1` means positive click and `0` means negative click
        points, labels = sample_points_mask_unified(mask)

        # # vis points
        # vis_points = draw_points(mask, points, labels)
        # cv2.imwrite('vis_sample_points.png', vis_points)

        frame_idx, object_ids, masks = predictor.add_new_points(state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,)

        # vis the mask of first frame
        print('masks:', masks.shape)
        tmp_mask = (masks[0] > 0.0).cpu().numpy()[0] # h,w
        cv2.imwrite('res_mask.png', (tmp_mask*255).astype(np.uint8))

        # propagate the prompts to get masklets throughout the video 
        video_segments = {}  # video_segments contains the per-frame segmentation results
        frames_mask = []
        frames_mask_vis = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            # print('out_frame_idx:', out_frame_idx)
            # print('out_obj_ids:', out_obj_ids)
            # print('out_mask_logits:', out_mask_logits.shape) # 1,1,h,w

            mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0] # h,w

            frame = frames[out_frame_idx]
            painted_image = mask_painter(frame, mask.astype('uint8'), mask_alpha=0.8)

            frames_mask.append((mask*255).astype(np.uint8))
            frames_mask_vis.append((painted_image).astype(np.uint8))

        predictor.reset_state(state)
        
        imageio.mimsave('res_occ.mp4', frames_mask, fps=30, quality=8, macro_block_size=1)
        imageio.mimsave('res_occ_vis.mp4', frames_mask_vis, fps=30, quality=8, macro_block_size=1)








