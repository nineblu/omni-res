import numpy as np
import torch
import cv2
import omni_res.models.utils.box_op as box_op

def process_box_labels(images, boxes, refs, seg_pesudo_label, seg_logit_teacher, det_label_q, seg_label_q, info_img, logger):
    mask_ratios = []
    logit_avgs = []
    for i in range(len(seg_pesudo_label)):
        h, w = seg_pesudo_label.shape[1], seg_pesudo_label.shape[2]
        box_x, box_y, box_w, box_h = det_label_q[i][0][0]*w, det_label_q[i][0][1]*h, det_label_q[i][0][2]*w, det_label_q[i][0][3]*h
        box_x, box_y, box_w, box_h = int(box_x-box_w/2), int(box_y-box_h/2), int(box_w), int(box_h)
        
        try:
            logit_avg = seg_logit_teacher[i][box_y:box_y+box_h,box_x:box_x+box_w].mean()
            if not torch.isnan(logit_avg):
                logit_avgs.append(logit_avg)

            mask_ratio = process_boxes_plus(seg_pesudo_label, box_x, box_y, box_w, box_h, i)
            if not torch.isnan(mask_ratio):
                mask_ratios.append(mask_ratio)
        except Exception:
            logger.warning('box outside image')
        
    return torch.Tensor(mask_ratios).mean(), torch.Tensor(logit_avgs).mean()

def process_boxes(seg_pesudo_label, box_x, box_y, box_w, box_h, i):
    box_index = torch.zeros((seg_pesudo_label.shape[1],seg_pesudo_label.shape[2])).bool()
    box_index[box_y:box_y+box_h,box_x:box_x+box_w] = True
    seg_pesudo_label[i][~box_index] = 0.0

def process_boxes_plus(seg_pesudo_label, box_x, box_y, box_w, box_h, i):
    process_boxes(seg_pesudo_label, box_x, box_y, box_w, box_h, i)
    mask_ratio = seg_pesudo_label[i].sum()/(box_h*box_w)
    if mask_ratio<0.2:
        seg_pesudo_label[i] = False
    return mask_ratio

def process_boxes_plus_v0(seg_pesudo_label, box_x, box_y, box_w, box_h, i):
    process_boxes(seg_pesudo_label, box_x, box_y, box_w, box_h, i)
    mask_ratio = seg_pesudo_label[i].sum()/(box_h*box_w)
    if mask_ratio<0.2:
        seg_pesudo_label[i] = False
    return mask_ratio


def process_point_labels(seg_pesudo_label, det_label_q, seg_label_q):
    gt_boxes = []
    mask_boxes = []
    for i in range(len(seg_pesudo_label)):
        h, w = seg_pesudo_label.shape[1], seg_pesudo_label.shape[2]
        non_zero_indices = torch.nonzero(seg_label_q[i][0])
        if non_zero_indices.shape[0] == 0:
            continue
        point_y = round(non_zero_indices[:, 0].float().mean().item())
        point_x = round(non_zero_indices[:, 1].float().mean().item())
        mask_box = process_points_v1(seg_pesudo_label, point_y, point_x, i)
        if len(mask_box)>0:
            box_x, box_y, box_w, box_h = det_label_q[i][0][0]*w, det_label_q[i][0][1]*h, det_label_q[i][0][2]*w, det_label_q[i][0][3]*h
            box_x, box_y, box_w, box_h = int(box_x-box_w/2), int(box_y-box_h/2), int(box_w), int(box_h)
            gt_boxes.append([box_x, box_y, box_x+box_w, box_y+box_h])
            mask_boxes.append([mask_box[0], mask_box[1], mask_box[0]+mask_box[2], mask_box[1]+mask_box[3]])
    
    if len(mask_boxes)==0:
        return torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
    
    ious = box_op.batch_box_iou_(torch.Tensor(mask_boxes), torch.Tensor(gt_boxes)).sum()/len(det_label_q)
    ious_5 = box_op.batch_box_iou(torch.Tensor(mask_boxes), torch.Tensor(gt_boxes)).sum()/len(det_label_q)
    ious_6 = box_op.batch_box_iou(torch.Tensor(mask_boxes), torch.Tensor(gt_boxes), threshold=0.6).sum()/len(det_label_q)
    ious_7 = box_op.batch_box_iou(torch.Tensor(mask_boxes), torch.Tensor(gt_boxes), threshold=0.7).sum()/len(det_label_q)
    return ious, ious_5, ious_6, ious_7

def process_points_v1(seg_pesudo_label, point_y, point_x, i):
    box = get_box_from_mask(seg_pesudo_label, point_y, point_x, i)
    box_index = torch.zeros((seg_pesudo_label.shape[1],seg_pesudo_label.shape[2])).bool()
    if len(box)==4:
        box_index[box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = True
    seg_pesudo_label[i][~box_index] = 0.0
    return box

def get_box_from_mask(seg_pesudo_label, point_y, point_x, i):
    gray = np.uint8(seg_pesudo_label[i].cpu().numpy()*255)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    max_box = []
    for bbox in bounding_boxes:
        [x, y, w, h] = bbox
        if y<point_y<y+h and x<point_x<x+w:
            if len(max_box)==0 or w*h>max_box[2]*max_box[3]:
                max_box = [x, y, w, h]
    return max_box

def process_point_distance(seg_pesudo_label, det_label_q, seg_logit_u):
    h, w = seg_pesudo_label.shape[1], seg_pesudo_label.shape[2]
    point_y, point_x = det_label_q[:, 0, 1]*h, det_label_q[:, 0, 0]*w
    seg_pesudo_label_xy = torch.Tensor([i for i in range(seg_pesudo_label.shape[1])])+0.5
    seg_pesudo_label_x = seg_pesudo_label_xy.unsqueeze(0).unsqueeze(0).repeat(seg_pesudo_label.shape[0], seg_pesudo_label.shape[1], 1)
    seg_pesudo_label_y = seg_pesudo_label_xy.unsqueeze(0).unsqueeze(-1).repeat(seg_pesudo_label.shape[0], 1, seg_pesudo_label.shape[1])

    mean_x = (seg_pesudo_label*seg_pesudo_label_x.to(seg_pesudo_label.device)).sum(dim=(1,2))/seg_pesudo_label.sum(dim=(1,2))
    mean_y = (seg_pesudo_label*seg_pesudo_label_y.to(seg_pesudo_label.device)).sum(dim=(1,2))/seg_pesudo_label.sum(dim=(1,2))

    index_x = ~torch.isnan(mean_x)
    index_y = ~torch.isnan(mean_y)
    index = torch.logical_and(index_x, index_y)

    return point_y[index], point_x[index], mean_y[index], mean_x[index]