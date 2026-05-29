import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from functools import partial

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x

def global_cosine_hm_percent(a, b, p=0.9, factor=0.1):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # Find threshold for bottom p% of well-reconstructed points
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    if len(f1s) == 0:
        return 0.0
    return f1s.max()

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    
    # Handle check for binary masks safely
    unique_vals = set(masks.flatten())
    assert unique_vals.issubset({0, 1}), "masks must be binary {0, 1}"
    if len(unique_vals) <= 1:
        return 0.0

    # Precompute region coords and areas, and extract their corresponding values from amaps
    region_vals = []
    region_areas = []
    for img_idx, mask in enumerate(masks):
        regions = measure.regionprops(measure.label(mask))
        for r in regions:
            coords = r.coords
            coords_y = coords[:, 0]
            coords_x = coords[:, 1]
            # Pre-extract the anomaly values for the region
            region_vals.append(amaps[img_idx, coords_y, coords_x])
            region_areas.append(r.area)

    # Pre-extract the values for all good pixels (where mask is 0)
    inverse_masks_bool = (masks == 0)
    amaps_flat_good = amaps[inverse_masks_bool]
    inverse_masks_sum = inverse_masks_bool.sum()

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    if delta == 0:
        return 0.0

    records = []
    for th in np.arange(min_th, max_th, delta):
        pros = []
        for vals, area in zip(region_vals, region_areas):
            tp_pixels = (vals > th).sum()
            pros.append(tp_pixels / area)

        fp_pixels = (amaps_flat_good > th).sum()
        fpr = fp_pixels / (inverse_masks_sum if inverse_masks_sum > 0 else 1)

        records.append({"pro": mean(pros) if pros else 0.0, "fpr": fpr, "threshold": th})

    df = pd.DataFrame(records)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    if len(df) == 0:
        return 0.0
    
    # Avoid division by zero if max FPR is 0
    max_fpr = df["fpr"].max()
    if max_fpr > 0:
        df["fpr"] = df["fpr"] / max_fpr
    else:
        df["fpr"] = 0.0

    from sklearn.metrics import auc
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def get_gaussian_kernel(kernel_size=5, sigma=4, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
                      )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels, bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def evaluation_batch_loco(model, dataloader, device, max_ratio=0.01, resize_mask=256):
    """
    Evaluates the model on MVTec LOCO AD dataset.
    Returns: [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, auroc_logic, auroc_struct, auroc_both]
    """
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    defect_type_list = []
    
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)

            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
                
            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map_flat = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
                
            pr_list_sp.append(sp_score)
            defect_type_list.extend(defect_type)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px_flat, pr_list_px_flat = gt_list_px.ravel(), pr_list_px.ravel()

        try:
            auroc_px = roc_auc_score(gt_list_px_flat, pr_list_px_flat) if len(np.unique(gt_list_px_flat)) > 1 else 0.0
        except Exception:
            auroc_px = 0.0

        try:
            auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp) if len(np.unique(gt_list_sp)) > 1 else 0.0
        except Exception:
            auroc_sp = 0.0

        try:
            ap_px = average_precision_score(gt_list_px_flat, pr_list_px_flat) if len(np.unique(gt_list_px_flat)) > 1 else 0.0
        except Exception:
            ap_px = 0.0

        try:
            ap_sp = average_precision_score(gt_list_sp, pr_list_sp) if len(np.unique(gt_list_sp)) > 1 else 0.0
        except Exception:
            ap_sp = 0.0

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px_flat, pr_list_px_flat)
        
        defect_type_list = np.array(defect_type_list)
        
        # Calculate LOCO-specific logical and structural AUROCs
        logic_mask = np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')
        struct_mask = np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')
        
        try:
            auroc_logic = roc_auc_score(gt_list_sp[logic_mask], pr_list_sp[logic_mask]) if len(np.unique(gt_list_sp[logic_mask])) > 1 else 0.0
        except Exception:
            auroc_logic = 0.0

        try:
            auroc_struct = roc_auc_score(gt_list_sp[struct_mask], pr_list_sp[struct_mask]) if len(np.unique(gt_list_sp[struct_mask])) > 1 else 0.0
        except Exception:
            auroc_struct = 0.0
        auroc_both = (auroc_logic + auroc_struct) / 2

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, auroc_logic, auroc_struct, auroc_both]

from torch.optim.lr_scheduler import _LRScheduler

class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for _ in self.base_lrs]
