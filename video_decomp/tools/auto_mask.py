import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import argparse
from loguru import logger

# # use bfloat16 for the entire notebook
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True



def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou

            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    # import ipdb; ipdb.set_trace()
    return selected_idx

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def masks_update_single(masks_lvl, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks    
    seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
    iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
    stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

    scores = stability * iou_pred
    keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
    masks_lvl = filter(keep_mask_nms, masks_lvl)

    return masks_lvl

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask(mask,frame_idx,save_dir):
    image_array = (mask * 255).astype(np.uint8)
    # 创建图像对象
    image = Image.fromarray(image_array[0])

    # 保存图像
    image.save(os.path.join(save_dir,f'{frame_idx:03}.png'))

def save_masks(mask_list,frame_idx,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    if len(mask_list[0].shape) == 3:
        # 计算拼接图片的尺寸
        total_width = mask_list[0].shape[2] * len(mask_list)
        max_height = mask_list[0].shape[1]
        # 创建大图片
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img[0] * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))
    else:
        # 计算拼接图片的尺寸
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        # 创建大图片
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))

def save_masks_npy(mask_list,frame_idx,save_dir):
    np.save(os.path.join(save_dir,f"mask_{frame_idx:03}.npy"),np.array(mask_list))
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def make_enlarge_bbox(origin_bbox, max_width,max_height,ratio):
    width = origin_bbox[2]
    height = origin_bbox[3]
    new_box = [max(origin_bbox[0]-width*(ratio-1)/2,0),max(origin_bbox[1]-height*(ratio-1)/2,0)]
    new_box.append(min(width*ratio,max_width-new_box[0]))
    new_box.append(min(height*ratio,max_height-new_box[1]))
    return new_box

def sample_points(masks, enlarge_bbox,positive_num=1,negtive_num=40):
    ex, ey, ewidth, eheight = enlarge_bbox
    positive_count = positive_num
    negtive_count = negtive_num
    output_points = []
    while True:
        x = int(np.random.uniform(ex, ex + ewidth))
        y = int(np.random.uniform(ey, ey + eheight))
        if masks[y][x]==True and positive_count>0:
            output_points.append((x,y,1))
            positive_count-=1
        elif masks[y][x]==False and negtive_count>0:
            output_points.append((x,y,0))
            negtive_count-=1
        if positive_count == 0 and negtive_count == 0:
            break

    return output_points

def sample_points_from_mask(mask):
    # 获取所有True值的索引
    true_indices = np.argwhere(mask)

    # 检查是否存在True值
    if true_indices.size == 0:
        raise ValueError("The mask does not contain any True values.")

    # 从True值索引中随机抽取一个点
    random_index = np.random.choice(len(true_indices))
    sample_point = true_indices[random_index]

    return tuple(sample_point)


def search_new_obj(masks_from_prev, mask_list,other_masks_list=None,mask_ratio_thresh=0,ratio=0.5, area_threash = 5000):
    new_mask_list = []

    # 计算mask_none，表示不包含在任何一个之前的mask中的区域
    mask_none = ~masks_from_prev[0].copy()[0]
    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~prev_mask[0]

    for mask in mask_list:
        seg = mask['segmentation']
        if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)
    
    for mask in new_mask_list:
        mask_none &= ~mask['segmentation']
    logger.info(len(new_mask_list))
    # import ipdb; ipdb.set_trace()
    logger.info("now ratio:",mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) )
    logger.info("expected ratios:",mask_ratio_thresh)
    if other_masks_list is not None:
        for mask in other_masks_list:
            if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh: # 还有很多的空隙，大于当前 thresh
                seg = mask['segmentation']
                if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
                    new_mask_list.append(mask)
                    mask_none &= ~seg
            else:
                break
    logger.info(len(new_mask_list))

    return new_mask_list

def get_bbox_from_mask(mask):
    # 获取非零元素的行列索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # 找到非零行和列的最小和最大索引
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # 计算宽度和高度
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    
    return xmin, ymin, width, height

def cal_no_mask_area_ratio(out_mask_list):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0].copy()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return(mask_none.sum() / (h * w))


class Prompts:
    def __init__(self,bs:int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        self.key_frame_obj_begin_list = []

    def add(self,obj_id,frame_id,mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False
        self.prompts[obj_id].append((frame_id,mask))
        if frame_id not in self.key_frame_list and new_obj:
            # import ipdb; ipdb.set_trace()
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
            logger.info("key_frame_obj_begin_list:",self.key_frame_obj_begin_list)
    
    def get_obj_num(self):
        return len(self.obj_list)
    
    def __len__(self):
        if self.obj_list % self.batch_size == 0:
            return len(self.obj_list) // self.batch_size
        else:
            return len(self.obj_list) // self.batch_size +1
    
    def __iter__(self):
        # self.batch_index = 0
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list)-1:
                end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
            else:
                if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]:
                    end_idx = self.start_idx+self.batch_size
                else:
                    end_idx =  self.key_frame_obj_begin_list[self.iter_frameindex+1]
                    self.iter_frameindex+=1
                # end_idx = min(self.start_idx+self.batch_size, self.key_frame_obj_begin_list[self.iter_frameindex+1])
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        # if self.batch_index * self.batch_size < len(self.obj_list):
        #     start_idx = self.batch_index * self.batch_size
        #     end_idx = min(start_idx + self.batch_size, len(self.obj_list))
        #     batch_keys = self.obj_list[start_idx:end_idx]
        #     batch_prompts = {key: self.prompts[key] for key in batch_keys}
        #     self.batch_index += 1
        #     return batch_prompts
        else:
            raise StopIteration
        
def get_video_segments(prompts_loader,predictor,inference_state,final_output=False):

    video_segments = {}
    for batch_prompts in tqdm(prompts_loader,desc="processing prompts\n"):
        predictor.reset_state(inference_state)
        for id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                # import ipdb; ipdb.set_trace()
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt[0],
                    obj_id=id,
                    mask=prompt[1]
                )
        # start_frame_idx = 0 if final_output else None
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = { }
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
        
        if final_output:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--level",choices=['default','small','middle','large'])
    parser.add_argument("--batch_size",type=int,default=20)
    parser.add_argument("--detect_stride",type=int,default=10)
    parser.add_argument("--use_other_level",type=int,default=1)
    parser.add_argument("--postnms",type=int,default=1)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    args = parser.parse_args()
    logger.add(os.path.join(args.output_dir,f'{args.level}.log'), rotation="500 MB")
    logger.info(args)
    video_dir = args.video_path
    level = args.level
    base_dir = args.output_dir

    ##### load Sam2 and Sam1 Model #####
    sam2_checkpoint = "../models/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)

    sam_ckpt_path="../models/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=args.pred_iou_thresh, 
        box_nms_thresh=args.box_nms_thresh, 
        stability_score_thresh=args.stability_score_thresh, 
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    print('frame_names:', frame_names)


    now_frame = 0
    # inference_state = predictor.init_state(video_path=video_dir)
    masks_from_prev = []
    sum_id = 0 # 记录一共有多少个物体

    prompts_loader = Prompts(bs=args.batch_size)  # hold all the clicks we add for visualization
    while True:
        logger.info(f"frame: {now_frame}")

        sum_id = prompts_loader.get_obj_num()
        image_path = os.path.join(video_dir,frame_names[now_frame])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
        if args.postnms:
            masks_default, masks_s, masks_m, masks_l = \
                masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        if level == 'default':
            masks = [mask for mask in masks_default]
            other_masks = [mask for mask in masks_l] + [mask for mask in masks_m] + [mask for mask in masks_s] 
        elif level == 'small':
            masks = [mask for mask in masks_s]
            other_masks = None
        elif level == 'middle':
            masks = [mask for mask in masks_m]
            other_masks = [mask for mask in masks_s]
        elif level == 'large':
            masks = [mask for mask in masks_l]
            other_masks = [mask for mask in masks_s] + [mask for mask in masks_m]
        else:
            raise NotImplementedError

        print('masks_default:', masks_default)
        print('masks_l:', masks_l)

        input_box = [0, 0, image.shape[1], image.shape[0]]
        masks_default_vis = show_automask_anno(masks_default, input_box, text_box_list=None)
        cv2.imwrite('vis_masks_default.png', masks_default_vis)

        masks_l_vis = show_automask_anno(masks_l, input_box, text_box_list=None)
        cv2.imwrite('vis_masks_l.png', masks_l_vis)

        masks_m_vis = show_automask_anno(masks_m, input_box, text_box_list=None)
        cv2.imwrite('vis_masks_m.png', masks_m_vis)

        masks_s_vis = show_automask_anno(masks_s, input_box, text_box_list=None)
        cv2.imwrite('vis_masks_s.png', masks_s_vis)



    #     if not args.use_other_level:
    #         other_masks = None
    #     if now_frame == 0: # first frame
    #         ann_obj_id_list = range(len(masks))

    #         save_masks([masks[ann_obj_id]['segmentation'] for ann_obj_id in ann_obj_id_list],now_frame,os.path.join(base_dir,level,'mask_each_frame-sam1'))
            
    #         if os.getenv("ONLY_STATISTIC","f") == 't':
    #             width = masks[0]['segmentation'].shape[1]
    #             height = masks[0]['segmentation'].shape[0]
    #             all_mask = np.zeros((height,width))
    #             for ann_obj_id in ann_obj_id_list:
    #                 all_mask = np.logical_or(all_mask,masks[ann_obj_id]['segmentation'])
                
    #             # import ipdb; ipdb.set_trace()
    #             img = Image.fromarray((all_mask * 255).astype(np.uint8)).convert("RGB")
    #             img.save(os.path.join(base_dir,level,'mask_each_frame-sam1','all.png'))
    #             logger.info(f"num:{len(ann_obj_id_list)}")
    #             logger.info(f"no mask ratio:{all_mask.sum()/(width*height)}")
    #             exit()

    #         for ann_obj_id in tqdm(ann_obj_id_list):
    #             seg = masks[ann_obj_id]['segmentation']
    #             prompts_loader.add(ann_obj_id,0,seg)

    #     else:  
    #         save_masks([mask['segmentation'] for mask in masks],now_frame,os.path.join(base_dir,level,'mask_each_frame-sam1'))
    #         new_mask_list = search_new_obj(masks_from_prev, masks, other_masks,mask_ratio_thresh)
    #         logger.info(f"number of new obj: {len(new_mask_list)}")

    #         for id,mask in enumerate(masks_from_prev):
    #             if mask.sum() == 0:
    #                 continue
    #             prompts_loader.add(id,now_frame,mask[0])

    #         for i in range(len(new_mask_list)):
    #             new_mask = new_mask_list[i]['segmentation']
    #             prompts_loader.add(sum_id+i,now_frame,new_mask)

    #     logger.info(f"obj num: {prompts_loader.get_obj_num()}")

    #     if now_frame==0 or len(new_mask_list)!=0:
    #         video_segments = get_video_segments(prompts_loader,predictor,inference_state)
    #     # video_segments contains the per-frame segmentation results
        
    #     vis_frame_stride = args.detect_stride
    #     plt.close("all")
    #     save_dir = os.path.join(base_dir,level,f"mask_each_frame_sam2")
    #     os.makedirs(save_dir,exist_ok=True)
    #     os.makedirs(os.path.join(save_dir,f"now_frame_{now_frame}"),exist_ok=True)
    #     max_area_no_mask = (0,-1)
    #     for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
    #         if out_frame_idx < now_frame:
    #             continue
    #         # 创建一个新的图形对象
    #         fig, ax = plt.subplots(figsize=(6, 4))
    #         ax.set_title(f"frame {out_frame_idx}")
            
    #         # 显示图像
    #         img_path = os.path.join(video_dir, frame_names[out_frame_idx])
    #         ax.imshow(Image.open(img_path))
    #         # 显示分割掩码
    #         out_mask_list = []
    #         for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #             idx_save_dir = os.path.join(save_dir,f"obj_{out_obj_id:02}")
    #             # os.makedirs(idx_save_dir,exist_ok=True)
    #             # import ipdb; ipdb.set_trace()
    #             show_mask(out_mask, ax, obj_id=out_obj_id,random_color=False)
    #             out_mask_list.append(out_mask)
            
    #         no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
    #         if now_frame == out_frame_idx:
    #             mask_ratio_thresh = no_mask_ratio
    #             logger.info(f"mask_ratio_thresh: {mask_ratio_thresh}")

    #         save_masks(out_mask_list, out_frame_idx,os.path.join(save_dir,f"now_frame_{now_frame}"))
    #         save_masks_npy(out_mask_list, out_frame_idx,os.path.join(save_dir,f"now_frame_{now_frame}"))
            
    #         # 保存图像
    #         plt.savefig(os.path.join(save_dir, f"frame_{out_frame_idx}.png"))
            
    #         # 关闭当前图形对象，释放内存
    #         plt.close(fig)
    #         if no_mask_ratio > mask_ratio_thresh + 0.01 and out_frame_idx > now_frame:
    #             masks_from_prev = out_mask_list
    #             max_area_no_mask = (no_mask_ratio, out_frame_idx)
    #             logger.info(max_area_no_mask)
    #             # mask_ratio_thresh = no_mask_ratio
    #             break
    #     if max_area_no_mask[1] == -1:
    #         break
    #     logger.info("max_area_no_mask:", max_area_no_mask)
    #     now_frame = max_area_no_mask[1]


    # ###### Final output ######
    # save_dir = os.path.join(base_dir,level,"final-output")
    # video_segments = get_video_segments(prompts_loader,predictor,inference_state,final_output=True)
    # for out_frame_idx in tqdm(range(0, len(frame_names), 1)):
    #     out_mask_list = []
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         out_mask_list.append(out_mask)
        
    #     # 显示图像
    #     img_path = os.path.join(video_dir, frame_names[out_frame_idx])
    #     ax.imshow(Image.open(img_path))

    #     no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
    #     logger.info(no_mask_ratio)

    #     save_masks(out_mask_list, out_frame_idx,save_dir)
    #     save_masks_npy(out_mask_list, out_frame_idx,save_dir)
    
