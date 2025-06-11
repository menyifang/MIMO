import numpy as np
import cv2
import glob
import imageio
from PIL import Image
import os

def all_file(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.png' or extend == '.jpg' or extend == '.jpeg' or extend == '.JPG' or extend == '.mp4':
                L.append(os.path.join(root, file))
    return L

def crop_img(img, mask):
    # find the bounding box
    x, y, w, h = cv2.boundingRect(mask) #91 85 554 1836
    y_max = y + h
    x_max = x + w
    # extend the bounding box with 0.1
    y = max(0, y - int(h * 0.05))
    y_max = min(img.shape[0], y_max + int(h * 0.05))
    return img[y:y_max, x:x_max]

def pad_img(img, color=[255, 255, 255]):
    # pad to square with mod 16 ==0
    h, w = img.shape[:2]
    max_size = max(h, w)
    if max_size % 16 != 0:
        max_size = int(max_size / 16) * 16 + 16
    top = (max_size - h) // 2
    bottom = max_size - h - top
    left = (max_size - w) // 2
    right = max_size - w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    padding_v = [top, bottom, left, right]
    return img, padding_v

def extract_mask_sdc(img):
    # >0 value as human
    mask = np.zeros_like(img[:, :, 0])
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # mask[gray[:, :] > 0] = 255
    mask[gray[:, :] > 10] = 255 # !!bug: remove noise
    return mask

def clean_mask(mask):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    return mask

def crop_img_sdc(img, mask):
    # find the bounding box
    x, y, w, h = cv2.boundingRect(mask) #91 85 554 1836
    y_max = y + h
    x_max = x + w
    # y = max(0, y-2)
    pad_h = 0.1
    pad_w = 0.05
    y = max(0, y - int(h * pad_h))
    y_max = min(img.shape[0], y_max + int(h * pad_h))
    x = max(0, x - int(w * pad_w))
    x_max = min(img.shape[1], x_max + int(w * pad_w))
    return y, y_max,x,x_max

def crop_human(pose_images, vid_images, mask_images):
    # find the bbox of the human in the whole frames
    bbox = []
    y = 10000
    y_max = 0
    x = 10000
    x_max = 0
    n_frame = len(pose_images)
    for pose_img in pose_images:
        frame = np.array(pose_img)
        mask = extract_mask_sdc(frame)
        y_, y_max_, x_, x_max_ = crop_img_sdc(frame, mask)
        y = min(y, y_)
        y_max = max(y_max, y_max_)
        x = min(x, x_)
        x_max = max(x_max, x_max_)
    # ensure width and height divisible by 2
    h = y_max - y
    w = x_max - x
    if h % 2 == 1:
        h += 1
        y_max += 1
    if w % 2 == 1:
        w += 1
        x_max += 1
    
    bbox = [x,x_max,y,y_max]

    # crop the human in the whole frames
    frames_res = []
    vid_res = []
    mask_res = []
    for i, pose_img in enumerate(pose_images):
        frame = np.array(pose_img)
        frame = frame[y:y_max, x:x_max]
        frame = Image.fromarray(frame)
        frames_res.append(frame)

        vid = vid_images[i]
        vid = np.array(vid)
        vid_res.append(Image.fromarray(vid[y:y_max, x:x_max]))

        mask = mask_images[i]
        mask = np.array(mask)
        mask_res.append(Image.fromarray(mask[y:y_max, x:x_max]))
    return frames_res, vid_res, mask_res


def init_bbox():
    return [10000, 0, 10000, 0]

def bbox_div2(x, x_max, y, y_max):
    # ensure width and height divisible by 2
    h = y_max - y
    w = x_max - x
    if h % 2 == 1:
        h += 1
        y_max += 1
    if w % 2 == 1:
        w += 1
        x_max += 1
    return x, x_max, y, y_max

def bbox_pad(x, x_max, y, y_max, img):
    w = x_max - x
    h = y_max - y
    # pad to square with mod 16 ==0
    max_size = max(h, w)
    if max_size % 16 != 0:
        max_size = int(max_size / 16) * 16 + 16
    top = (max_size - h) // 2
    bottom = max_size - h - top
    left = (max_size - w) // 2
    right = max_size - w - left

    y = max(0, y-top)
    y_max = min(img.shape[0], y_max+bottom)
    x = max(0, x-left)
    x_max = min(img.shape[1], x_max+right)

    return x, x_max, y, y_max

def compute_area_ratio(bbox_frame, bbox_clip):
    x1, x2, y1, y2 = bbox_frame
    x1_clip, x2_clip, y1_clip, y2_clip = bbox_clip
    area_frame = (x2 - x1) * (y2 - y1)
    area_clip = (x2_clip - x1_clip) * (y2_clip - y1_clip)
    ratio = area_frame / area_clip
    return ratio

def update_clip(bbox_clip, start_idx, i, bbox_max):
    x, x_max, y, y_max = bbox_max
    for j in range(start_idx, i):
        bbox_clip[j] = [x, x_max, y, y_max]

def crop_human_clip_auto_context(pose_images, vid_images, bk_images, overlay=4):
    # find the bbox of the human in the clip frames
    bbox_clip = []
    bbox_perframe = []
    ratio_list = []
    x, x_max, y, y_max = init_bbox()
    n_frame = len(pose_images)

    context_list = []
    bbox_clip_list = []

    areas = np.zeros(n_frame)
    start_idx = 0
    for i in range(0, n_frame):
        # print('i:', i)
        pose_img = pose_images[i]
        frame = np.array(pose_img)
        mask = extract_mask_sdc(frame)
        mask = clean_mask(mask)
        y_, y_max_, x_, x_max_ = crop_img_sdc(frame, mask)
        x_, x_max_, y_, y_max_ = bbox_div2(x_, x_max_, y_, y_max_)
        x_, x_max_, y_, y_max_ = bbox_pad(x_, x_max_, y_, y_max_, frame)
        bbox_max_prev = (x, x_max, y, y_max)

        # update max
        y = min(y, y_)
        y_max = max(y_max, y_max_)
        x = min(x, x_)
        x_max = max(x_max, x_max_)
        bbox_max_cur = (x, x_max, y, y_max)

        # save bbox per frame
        bbox_cur = [x_, x_max_, y_, y_max_]
        bbox_perframe.append(bbox_cur)
        bbox_clip.append(bbox_cur)

        # compute the area of each frame
        area = (x_max_ - x_) * (y_max_ - y_)/100
        areas[i] = area
        area_max = (y_max - y) * (x_max - x)/100
        if area_max!=0:
            ratios = areas[start_idx:i]/area_max
        else:
            ratios = np.zeros(i-start_idx)

        # ROI_THE = 0.2
        ROI_THE = 0.5
        if (i == n_frame - 1):
            i += 1
            # print('update from ')
            # print('start_idx:', start_idx)
            # print('i:', i)

            # print('clip from to:', range(start_idx, i))
            if len(context_list)==0:
                context_list.append(list(range(start_idx, i)))
            else:
                overlay_ = min(overlay, len(context_list[-1]))
                context_list.append(list(range(start_idx-overlay_, i)))
            bbox_clip_list.append(bbox_max_cur)

            update_clip(bbox_clip, start_idx, i, bbox_max_cur)
            start_idx = i
            continue
        elif np.any(ratios < ROI_THE) and ratios.sum()!=0:

            # generate a list from start_idx to i
            if len(context_list)==0:
                context_list.append(list(range(start_idx, i)))
            else:
                overlay_ = min(overlay, len(context_list[-1]))
                context_list.append(list(range(start_idx-overlay_, i)))
            bbox_clip_list.append(bbox_max_prev)

            # print('update from ')
            # print('start_idx:', start_idx)
            # print('i:', i)
            update_clip(bbox_clip, start_idx, i, bbox_max_prev)
            x, x_max, y, y_max = bbox_cur
            start_idx = i
            continue

    # vis ratio
    for i in range(0, n_frame):
        # print('i:', i)
        bbox_frame_ = bbox_perframe[i]
        bbox_clip_ = bbox_clip[i]
        # print('bbox_frame_:', bbox_frame_)
        # print('bbox_clip_:', bbox_clip_)
        if np.array(bbox_clip_).sum()==0:
            ratio = 0
        else:
            ratio = compute_area_ratio(bbox_frame_, bbox_clip_)
        # print('ratio:', ratio)
        ratio_list.append(ratio)

    # crop images
    frames_res = []
    vid_res = []
    bk_res = []
    for k, context in enumerate(context_list):
        for i in context:
            pose_img = pose_images[i]
            frame = np.array(pose_img)
            x, x_max, y, y_max = bbox_clip_list[k]
            if x >= x_max or y >= y_max:
                x, x_max, y, y_max = 0, frame.shape[1] - 1, 0, frame.shape[0] - 1
            frame = frame[y:y_max, x:x_max]
            frame = Image.fromarray(frame)
            frames_res.append(frame)

            vid = vid_images[i]
            vid = np.array(vid)
            vid_res.append(Image.fromarray(vid[y:y_max, x:x_max]))

            bk = bk_images[i]
            bk = np.array(bk)
            bk_res.append(Image.fromarray(bk[y:y_max, x:x_max]))

    return frames_res, vid_res, bk_res, bbox_clip, context_list, bbox_clip_list


def crop_human_clip(pose_images, vid_images, bk_images, clip_length=1):
    # find the bbox of the human in the clip frames
    bbox_clip = []
    x, x_max, y, y_max = init_bbox()
    n_frame = len(pose_images)
    for i in range(0, n_frame):
        # print('i:', i)
        pose_img = pose_images[i]
        frame = np.array(pose_img)
        mask = extract_mask_sdc(frame)
        mask = clean_mask(mask)
        y_, y_max_, x_, x_max_ = crop_img_sdc(frame, mask)
        x_, x_max_, y_, y_max_ = bbox_div2(x_, x_max_, y_, y_max_)
        x_, x_max_, y_, y_max_ = bbox_pad(x_, x_max_, y_, y_max_, frame)
        
        # print(x_,x_max_,y_,y_max_)

        y = min(y, y_)
        y_max = max(y_max, y_max_)
        x = min(x, x_)
        x_max = max(x_max, x_max_)
        # print(x,x_max,y,y_max)

        if ((i+1) % clip_length == 0) or (i==n_frame-1):
            x, x_max, y, y_max = bbox_div2(x, x_max, y, y_max)
            if x>=x_max or y>=y_max:
                x, x_max, y, y_max = 0, frame.shape[1]-1, 0, frame.shape[0]-1
            # print(x,x_max,y,y_max)
            bbox_clip.append([x, x_max, y, y_max])
            x, x_max, y, y_max = init_bbox()
    # crop images
    frames_res = []
    vid_res = []
    bk_res = []
    for i, pose_img in enumerate(pose_images):
        x, x_max, y, y_max = bbox_clip[i//clip_length]
        frame = np.array(pose_img)
        frame = frame[y:y_max, x:x_max]
        frame = Image.fromarray(frame)
        frames_res.append(frame)

        vid = vid_images[i]
        vid = np.array(vid)
        vid_res.append(Image.fromarray(vid[y:y_max, x:x_max]))

        bk = bk_images[i]
        bk = np.array(bk)
        bk_res.append(Image.fromarray(bk[y:y_max, x:x_max]))
    return frames_res, vid_res, bk_res, bbox_clip


def init_bk(n_frame,h,w):
    images = []
    for i in range(n_frame):
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        images.append(Image.fromarray(img))
    return images



def pose_adjust(pose_image, width=512, height=784):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # PIL to numpy
    pose_img = np.array(pose_image)
    h, w, c = pose_img.shape
    # print('pose_img:', pose_img.shape)
    # resize
    # pose_img = cv2.resize(pose_img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    nh, nw = height, int(w * height / h)
    pose_img = cv2.resize(pose_img, (nw, nh), interpolation=cv2.INTER_AREA)
    if nw < width:
        # pad
        pad = (width - nw) // 2
        canvas[:, pad:pad + nw, :] = pose_img
    else:
        # center crop
        crop = (nw - width) // 2
        canvas = pose_img[:, crop:crop + width, :]

    # numpy to PIL
    canvas = Image.fromarray(canvas)
    return canvas


def load_pretrain_pose_guider(model, ckpt_path):

    state_dict = torch.load(ckpt_path, map_location="cpu")
    # for k,v in state_dict.items():
        # print(k, v.shape)

    weights = state_dict['conv_in.weight']
    # _,c,_,_ = weights.shape
    # if c!=
    weights = torch.cat((weights, torch.zeros_like(weights), torch.zeros_like(weights)), dim=1)
    state_dict['conv_in.weight'] = weights

    model.load_state_dict(state_dict, strict=True)

    return model

def refine_img_prepross(image, mask):
    im_ary = np.asarray(image).astype(np.float32)
    input = np.concatenate([im_ary, mask[:, :, np.newaxis]], axis=-1)
    return input

mask_mode = {'up_down_left_right': 0, 'left_right_up': 1, 'left_right_down': 2, 'up_down_left': 3, 'up_down_right': 4,
            'left_right': 5, 'up_down': 6, 'left_up': 7, 'right_up': 8, 'left_down': 9, 'right_down': 10,
             'left': 11, 'right': 12, 'up': 13, 'down': 14, 'inner': 15}

def get_mask(mask_list, bbox, img):
    w, h = img.size
    # print('size w h:', w, h)
    # print('bbox:', bbox)
    w_min, w_max, h_min, h_max = bbox
    if w_min<=0 and w_max>=w and h_min<=0 and h_max>=h: # up_down_left_right
        mode = 'up_down_left_right'
    elif w_min<=0 and w_max>=w and h_min<=0:
        mode = 'left_right_up'
    elif w_min<=0 and w_max>=w and h_max>=h:
        mode = 'left_right_down'
    elif w_min <= 0 and h_min <= 0 and h_max >= h:
        mode = 'up_down_left'
    elif w_max >= w and h_min <= 0 and h_max >= h:
        mode = 'up_down_right'

    elif w_min<=0 and w_max>=w: #
        mode = 'left_right'
    elif h_min<=0 and h_max>=h: #
        mode = 'up_down'
    elif w_min<=0 and h_min<=0: # left_up
        mode = 'left_up'
    elif w_max>=w and h_min<=0: # right_up5
        mode = 'right_up'
    elif w_min<=0 and h_max>=h: # left_down6
        mode = 'left_down'
    elif w_max>=w and h_max>=h: # right_down7
        mode = 'right_down'

    elif w_min<=0:
        mode = 'left'
    elif w_max>=w:
        mode = 'right'
    elif h_min<=0:
        mode = 'up'
    elif h_max>=h:
        mode = 'down'
    else:
        mode = 'inner'

    mask = mask_list[mask_mode[mode]]

    return mask

def load_mask_list(mask_path):
    mask_list = []
    for key in mask_mode.keys():
        mask = cv2.imread(mask_path[:-4] + '_%s.png'%key)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask_list.append(mask)
    return mask_list

def recover_bk(images, start_idx, end_idx, template_name=None):
    img = np.array(images[0])
    for i in range(start_idx, end_idx):
        if template_name == "dance_indoor_1":
            images[i][:img.shape[0], :, 0] = 255
            images[i][:img.shape[0], :, 1] = 255
            images[i][:img.shape[0], :, 2] = 255
        else:
            img_blank = np.ones_like(img) * 255
            images[i] = Image.fromarray(img_blank)
    return images


def load_video_fixed_fps(vid_path, target_fps=30, target_speed=1):
    # Load video and get metadata
    reader = imageio.get_reader(vid_path)
    fps = round(reader.get_meta_data()['fps'])
    # print('original fps:', fps)
    # print('target fps:', target_fps)

    # Calculate the ratio of original fps to target fps to determine which frames to keep
    keep_ratio = target_speed * fps / target_fps
    n_frames = reader.count_frames()
    keep_frames_indices = np.arange(0, n_frames, keep_ratio).astype(int)

    # Extract frames at the target frame rate
    frames = [Image.fromarray(reader.get_data(i)) for i in keep_frames_indices if i < len(reader)]        

    reader.close()
    return frames

    