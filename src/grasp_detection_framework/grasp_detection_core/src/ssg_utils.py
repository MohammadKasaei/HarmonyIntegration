import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from itertools import product
from math import sqrt
import numpy as np

from skimage.filters import gaussian
from skimage.feature import peak_local_max

import cv2


def encode(matched, priors):
    variances = [0.1, 0.2]

    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]  # 10 * (Xg - Xa) / Wa
    g_cxcy /= (variances[0] * priors[:, 2:])  # 10 * (Yg - Ya) / Ha
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 5 * log(Wg / Wa)
    g_wh = torch.log(g_wh) / variances[1]  # 5 * log(Hg / Ha)
    # return target for smooth_l1_loss

    offsets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]

    return offsets


def sanitize_coordinates(_x1, _x2, img_size, padding=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def sanitize_coordinates_numpy(_x1, _x2, img_size, padding=0):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, a_min=0, a_max=1000000)
    x2 = np.clip(x2 + padding, a_min=0, a_max=img_size)

    return x1, x2


def box_iou(box_a, box_b):
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    (n, A), B = box_a.shape[:2], box_b.shape[1]
    # add a dimension
    box_a = box_a[:, :, None, :].expand(n, A, B, 4)
    box_b = box_b[:, None, :, :].expand(n, A, B, 4)

    max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
    min_xy = torch.max(box_a[..., :2], box_b[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    out = inter_area / (area_a + area_b - inter_area)
    return out if use_batch else out.squeeze(0)


def match(cfg, box_gt, anchors, class_gt):
    # Convert prior boxes to the form of [xmin, ymin, xmax, ymax].
    decoded_priors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)

    overlaps = box_iou(box_gt, decoded_priors)  # (num_gts, num_achors)

    _, gt_max_i = overlaps.max(1)  # (num_gts, ) the max IoU for each gt box
    each_anchor_max, anchor_max_i = overlaps.max(0)  # (num_achors, ) the max IoU for each anchor

    # For the max IoU anchor for each gt box, set its IoU to 2. This ensures that it won't be filtered
    # in the threshold step even if the IoU is under the negative threshold. This is because that we want
    # at least one anchor to match with each gt box or else we'd be wasting training data.
    each_anchor_max.index_fill_(0, gt_max_i, 2)

    # Set the index of the pair (anchor, gt) we set the overlap for above.
    for j in range(gt_max_i.size(0)):
        anchor_max_i[gt_max_i[j]] = j

    anchor_max_gt = box_gt[anchor_max_i]  # (num_achors, 4)
    # For OCDI dataset
    conf = class_gt[anchor_max_i]  # the class of the max IoU gt box for each anchor
    # Others
    # conf = class_gt[anchor_max_i] + 1  # the class of the max IoU gt box for each anchor
    conf[each_anchor_max < cfg.pos_iou_thre] = -1  # label as neutral
    conf[each_anchor_max < cfg.neg_iou_thre] = 0  # label as background

    offsets = encode(anchor_max_gt, anchors)

    return offsets, conf, anchor_max_gt, anchor_max_i


def crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def ones_crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    # ones = torch.ones_like(masks).float().cuda()
    ones = torch.ones_like(masks).float().to(masks.device)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down
    out_mask = ~crop_mask

    return masks * crop_mask.float() + ones * out_mask.float()



def make_anchors(cfg, conv_h, conv_w, scale):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in cfg.aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / cfg.img_size
            h = scale / ar / cfg.img_size

            prior_data += [x, y, w, h]

    return prior_data



norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)


def fast_gr_nms(box_thre, coef_thre, gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, class_thre, cfg):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.size()
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_pos_coef_thre = gr_pos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_sin_coef_thre = gr_sin_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_cos_coef_thre = gr_cos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_wid_coef_thre = gr_wid_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]
    pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms = gr_pos_coef_thre[keep], gr_sin_coef_thre[keep], gr_cos_coef_thre[keep], gr_wid_coef_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]
    pos_coef_nms = pos_coef_nms[idx]
    sin_coef_nms = sin_coef_nms[idx]
    cos_coef_nms = cos_coef_nms[idx]
    wid_coef_nms = wid_coef_nms[idx]

    return box_nms, coef_nms, pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms, class_ids, class_nms




def gr_nms_v2(
    class_pred, box_pred, 
    coef_pred, proto_out, 
    gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
    anchors, cfg):

    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    gr_pos_coef_p = gr_pos_coef_pred.squeeze()
    gr_sin_coef_p = gr_sin_coef_pred.squeeze()
    gr_cos_coef_p = gr_cos_coef_pred.squeeze()
    gr_wid_coef_p = gr_wid_coef_pred.squeeze()


    if isinstance(anchors, list):
        anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

    class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]
    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]
    gr_pos_coef_thre = gr_pos_coef_p[keep, :]
    gr_sin_coef_thre = gr_sin_coef_p[keep, :]
    gr_cos_coef_thre = gr_cos_coef_p[keep, :]
    gr_wid_coef_thre = gr_wid_coef_p[keep, :]


    # decode boxes
    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    box_thre = torch.clip(box_thre, min=0., max=1.)

    if class_thre.shape[1] == 0:
        return None, None, None, None, None, None, None, None, None
    else:
        box_thre, coef_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, class_ids, class_thre = fast_gr_nms(
            box_thre, coef_thre, 
            gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, 
            class_thre, cfg
        )

        return class_ids, class_thre, box_thre, coef_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, proto_p


def detect_grasps(pos_masks, ang_masks, wid_masks, cls_ids, min_distance=2, threshold_abs=0.5, num_peaks=5, per_object_width=None):
    if per_object_width is None:
        from ssg_config import PER_CLASS_MAX_GRASP_WIDTH
    else:
        PER_CLASS_MAX_GRASP_WIDTH = per_object_width

    assert cls_ids.shape[0] == pos_masks.shape[0] == ang_masks.shape[0] == wid_masks.shape[0]
    grasps = []
    for i in range(cls_ids.shape[0]):
        tmp = []
        cls_id = cls_ids[i]-1
        max_width = PER_CLASS_MAX_GRASP_WIDTH[cls_id]

        pos_mask = np.array(pos_masks[i], dtype='float')
        
        local_max = peak_local_max(pos_mask, min_distance=min_distance, threshold_abs=threshold_abs, num_peaks=num_peaks)

        for p_array in local_max:
            grasp_point = tuple(p_array)
            grasp_angle = ang_masks[i][grasp_point] / np.pi * 180
            # if grasp_angle > 0:
            #     grasp_angle = grasp_angle - 90
            # elif grasp_angle < 0:
            #     grasp_angle = grasp_angle + 90
            grasp_width = wid_masks[i][grasp_point]
            tmp.append([float(grasp_point[1]), float(grasp_point[0]), grasp_width*max_width, 20, grasp_angle, int(cls_ids[i])])

        grasps.append(tmp)
    
    return grasps


def gr_post_processing(img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h, ori_w, num_grasp_per_object=1, per_object_width=None):
    keep = (class_p >= 0.3)
    if not keep.any():
        print("No valid instance")
    else:
        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]   
        pos_coef_p = pos_coef_p[keep]
        sin_coef_p = sin_coef_p[keep]
        cos_coef_p = cos_coef_p[keep]
        wid_coef_p = wid_coef_p[keep]
    
    ids_p = (ids_p + 1)
    ids_p = ids_p.cpu().numpy()
    box_p = box_p

    instance_masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t())).contiguous()
    # print("Instance masks: ", instance_masks.shape)
    # vis_masks = (instance_masks.clone().cpu().numpy()[:,:,-1] * 255).astype('uint8')
    # print(vis_masks.shape)
    # vis_masks = cv2.applyColorMap(vis_masks, cv2.COLORMAP_WINTER)
    # cv2.imwrite("results/images/vis_masks.png", vis_masks)
    instance_masks = crop(instance_masks, box_p).permute(2,0,1)


    pos_masks = torch.sigmoid(torch.matmul(proto_p, pos_coef_p.t())).contiguous()
    pos_masks = crop(pos_masks, box_p).permute(2,0,1)

    sin_masks = torch.matmul(proto_p, sin_coef_p.t()).contiguous()
    sin_masks = crop(sin_masks, box_p).permute(2,0,1)

    cos_masks = torch.matmul(proto_p, cos_coef_p.t()).contiguous()
    cos_masks = crop(cos_masks, box_p).permute(2,0,1)

    wid_masks = torch.sigmoid(torch.matmul(proto_p, wid_coef_p.t())).permute(2,0,1).contiguous()
    # wid_masks = crop(wid_masks, box_p).permute(2,0,1)
    wid_masks = wid_masks * pos_masks

    instance_masks = F.interpolate(instance_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    instance_masks.gt_(0.5)
    pos_masks = F.interpolate(pos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    sin_masks = F.interpolate(sin_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    cos_masks = F.interpolate(cos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    wid_masks = F.interpolate(wid_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    
    # Convert processed image to original size

    img = cv2.resize(img, (ori_w, ori_w))
    depth = cv2.resize(depth, (ori_w, ori_w))

    ori_img = img[0:ori_h, 0:ori_w, :]
    ori_img = ori_img * norm_std + norm_mean
    ori_depth = depth[0:ori_h, 0:ori_w]
    instance_masks = instance_masks[:, 0:ori_h, 0:ori_w]
    pos_masks = pos_masks[:, 0:ori_h, 0:ori_w]
    sin_masks = sin_masks[:, 0:ori_h, 0:ori_w]
    cos_masks = cos_masks[:, 0:ori_h, 0:ori_w]
    wid_masks = wid_masks[:, 0:ori_h, 0:ori_w]

    box_p = box_p.cpu().numpy()
    instance_masks = instance_masks.cpu().numpy()
    pos_masks = pos_masks.cpu().numpy()
    wid_masks = wid_masks.cpu().numpy()

    ang_masks = []

    for i in range(pos_masks.shape[0]):
        pos_masks[i] = gaussian(pos_masks[i], 2.0, preserve_range=True)
        # ang_masks[i] = gaussian(ang_masks[i], 2.0, preserve_range=True)
        # wid_masks[i] = gaussian(wid_masks[i], 1.0, preserve_range=True)
        ang_mask = (torch.atan2(sin_masks[i], cos_masks[i]) / 2.0).cpu().numpy().squeeze()
        ang_masks.append(ang_mask)

    ang_masks = np.array(ang_masks)


    scale = np.array([ori_w, ori_w, ori_w, ori_w])
    box_p *= scale
    box_p = np.concatenate([box_p, ids_p.reshape(-1,1)], axis=-1)

    grasps = detect_grasps(pos_masks, ang_masks, wid_masks, ids_p, num_peaks=num_grasp_per_object, per_object_width=per_object_width)


    return ori_img, ori_depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p

