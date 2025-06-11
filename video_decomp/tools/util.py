import imageio
import numpy as np
import os
import cv2
import torch
from PIL import Image

def all_file(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.mp4' or extend == '.MP4' or extend == '.mov' or extend == '.MOV':
                L.append(os.path.join(root, file))
    return L

def load_video_fixed_fps(vid_path, target_fps=30, target_speed=1):
    # Load video and get metadata
    reader = imageio.get_reader(vid_path)
    fps = round(reader.get_meta_data()['fps'])
    print('original fps:', fps)
    # Calculate the ratio of original fps to target fps to determine which frames to keep
    keep_ratio = target_speed * fps / target_fps
    n_frames = reader.count_frames()
    keep_frames_indices = np.arange(0, n_frames, keep_ratio).astype(int)

    # Extract frames at the target frame rate
    frames = [reader.get_data(i) for i in keep_frames_indices if i < len(reader)]

    reader.close()
    return frames

def get_main_person(boxes, frame):
    # get the main person closest to the center
    ## bbox: x0, y0, x1, y1
    center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])  # (x, y)
    boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
    dists = np.linalg.norm(boxes_center - center, axis=1)
    # print('dists:', dists)
    main_id = np.argmin(dists)
    return main_id

# def get_main_person(boxes, frame):
#     # get the main person closest to the center
#     ## bbox: x0, y0, x1, y1
#     center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])  # (x, y)
#     boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
#     dists = np.linalg.norm(boxes_center - center, axis=1)
#     main_id = np.argmin(dists)

#     return main_id

def clean_mask(mask):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    return mask

def get_bbox_from_mask(mask, img, pad_h = 0.01, pad_w = 0.01):
    # find the bounding box
    x, y, w, h = cv2.boundingRect(mask)  # 91 85 554 1836
    y_max = y + h
    x_max = x + w
    # y = max(0, y-2)
    # pad_h = 0.05
    # pad_w = 0.05
    # pad_h = 0.01
    # pad_w = 0.01
    y = max(0, y - int(h * pad_h))
    y_max = min(img.shape[0], y_max + int(h * pad_h))
    x = max(0, x - int(w * pad_w))
    x_max = min(img.shape[1], x_max + int(w * pad_w))
    bbox = np.zeros((1, 4))
    bbox[:, 0] = x
    bbox[:, 1] = y
    bbox[:, 2] = x_max
    bbox[:, 3] = y_max

    return bbox


def sample_points_mask_unified(mask):
    '''
    Sample points on the mask with valid pixels
    Args:
        mask:
    Returns:
        sample_points: np.array([[x1, y1],[x2,y2],...], dtype=np.float32)
    '''
    # get points'coordinate with valid pixels
    points = np.argwhere(mask > 0)  # h_idx,w_idx
    # unified sample n points
    sample_points = []
    labels = []
    # uniform sample n_points from points
    n_points = 10
    idxs = np.linspace(0, points.shape[0] - 1, n_points).astype(int)
    for idx in idxs:
        sample_points.append((points[idx][1], points[idx][0]))
        labels.append(1)
    # convert to np.array
    sample_points = np.array(sample_points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return sample_points, labels

def draw_points(mask, points, labels):
    # draw points with labels
    if len(mask.shape) == 2:
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

def vis_segment(color, alpha):
    bk = np.ones_like(color)*255
    color = color * alpha[:, :, np.newaxis] + bk * (1 - alpha[:, :, np.newaxis])
    color = color.astype(np.uint8)
    return color

def detect_adapt(frame, detector):
    print('detect_adapt frame:', frame.shape)
    img_cv2 = frame[:, :, ::-1]
    # Detect humans in image
    det_out = detector(img_cv2)
    # print('det_out:', det_out)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    return boxes

def compute_wrist_local_pose(root_global, tree_rot_mats, hand_global_orient):
    """ root_global：bsx3x3 rotation matrix
        tree_rot_mats: list of bsx3x3 rotation matrices
        hand_global_orient: bsx3x3 rotation matrix
    """
    # print('root_global:', root_global.shape)
    # print('tree_rot_mats:', tree_rot_mats.shape)
    # print('hand_global_orient:', hand_global_orient.shape)
    for i in range(tree_rot_mats.shape[1]):
        rot_mat = tree_rot_mats[:, i, :, :]
        root_global = root_global @ rot_mat
    # print('root_global:', root_global.shape)
    # root_global: bsx3x3
    # hand_global_orient: bsx3x3
    # multiply the inverse of root_global to hand_global_orient
    wrist_local = torch.matmul(root_global.transpose(1, 2), hand_global_orient)
    # print('wrist_local:', wrist_local.shape)
    return wrist_local

import trimesh
def load_ply(path):
    mesh = trimesh.load(path)
    vertices = mesh.vertices
    faces = mesh.faces
    return vertices, faces


def repair_faces(vertices, faces):
    max = faces.max()
    len = vertices.shape[0]
    if max == len:
        faces -= 1
    return faces


def apply_face_mask(mesh, face_mask):
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def apply_vertex_mask(mesh, vertex_mask):
    faces_mask = vertex_mask[mesh.faces].any(dim=1)
    mesh = apply_face_mask(mesh, faces_mask)
    return mesh


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H, W = int(H//2*2), int(W//2*2)
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def resize_video(frames):
    resolution = 720
    out_H, out_W, _ = frames[0].shape
    if out_H > resolution and out_W > resolution:
        for i in range(len(frames)):
            frames[i] = resize_image(frames[i], resolution)
    return frames

def get_clip_bbox(bbox_list):
    y = 10000
    y_max = 0
    x = 10000
    x_max = 0
    for bbox in bbox_list:
        x_, y_, x_max_, y_max_ = bbox['bbox'][0]
        y = min(y, y_)
        y_max = max(y_max, y_max_)
        x = min(x, x_)
        x_max = max(x_max, x_max_)

    x, y, x_max, y_max = int(x), int(y), int(x_max), int(y_max)

    bbox_clip = [x, y, x_max, y_max]
    return bbox_clip

def crop_human(vid_images, mask_images, bbox):
    x, y, x_max, y_max = bbox
    # # ensure width and height divisible by 2
    # h = y_max - y
    # w = x_max - x
    # if h % 2 == 1:
    #     h += 1
    #     y_max += 1
    # if w % 2 == 1:
    #     w += 1
    #     x_max += 1

    bbox = [x, y, x_max, y_max]

    # crop the human in the whole frames
    vid_res = []
    mask_res = []
    for i, vid in enumerate(vid_images):
        vid = np.array(vid)
        vid_res.append(Image.fromarray(vid[y:y_max, x:x_max]))

        mask = mask_images[i]
        mask = np.array(mask)
        mask_res.append(Image.fromarray(mask[y:y_max, x:x_max]))
    return vid_res, mask_res, bbox

def extract_mask_sdc(img):
    # >0 value as human
    mask = np.zeros_like(img[:, :, 0])
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # mask[gray[:, :] > 0] = 255
    mask[gray[:, :] > 10] = 1 # !!bug: remove noise
    return mask

def clean_mask(mask):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    return mask

def get_occ_frame(frames_mask, num_frame=5, interval=20):
    # input: frames_mask, list of mask frames; num_frame, number of frames to be considered with occlusion
    # output: occ_idxs, list of occ frames
    num_comp_list = []
    for idx, frame in enumerate(frames_mask):
        if len(frame.shape) == 3:
            frame = frame[..., 0]
        if np.max(frame)<=1:
            frame = (frame*255)
        
        # print('frame:', np.max(frame))
        # print(frame.shape)
        # print(frame)
        frame = clean_mask(frame)
        num_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(frame, connectivity=8)
        num_comp = num_comp - 1
        num_comp_list.append(num_comp)

    occ_idxs = np.argsort(num_comp_list)[::-1]
    # interval = 20
    occ_idxs_new = []
    for idx in occ_idxs:
        if len(occ_idxs_new)>=num_frame:
            break
        if len(occ_idxs_new) == 0:
            occ_idxs_new.append(idx)
        else:
            repeat = False
            for val in occ_idxs_new:
                if abs(val - idx) < interval:
                    repeat = True
                    break
            if not repeat:
                occ_idxs_new.append(idx)
    occ_idxs = sorted(occ_idxs_new)
    return occ_idxs

def get_valid_person(kps, threshold=0.3):
    key_joint = [0,1,2,5,8,9,10,11,12,13,14,15,16,17]
    score = kps[:, key_joint, 2]
    valid_person_ind = np.min(score, axis=-1) > threshold
    valid_person_ind = np.where(valid_person_ind)[0]

    return valid_person_ind

