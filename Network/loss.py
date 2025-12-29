import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
import sys
sys.path.append("/workspace/code/RadarSimulator2")
from Network.modules import PointNet2MSG
import math
#import MinkowskiEngine as ME
import scipy.spatial

def build_voxel_sample_masks(target, seed=2025):
    """
    构建稀疏 occupancy 监督中的近邻/远离负样本采样 mask（无 batch 循环）。

    Args:
        target: (B, D, H, W) occupancy ground truth, ∈ {0,1}
        seed: int, 随机采样的固定种子（保证每次采样一致）

    Returns:
        near_mask: (B, D, H, W) 靠近正样本的负样本
        far_mask: (B, D, H, W) 远离正样本的负样本
    """
    B, _, D, H, W = target.shape
    device = target.device

    # 设置固定随机种子
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # 正负样本 mask
    pos_mask = target > 0          # (B, D, H, W)
    neg_mask = target == 0

    # 正样本区域 dilation 得到邻近区域
    pos_mask_f = pos_mask.float()  # (B,1,D,H,W)
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    neighbor_prob = F.conv3d(pos_mask_f, kernel, padding=1)
    neighbor_mask = (neighbor_prob > 0) & neg_mask  # (B,1,D,H,W)
    far_mask = neg_mask & (~neighbor_mask)

    # 展平
    pos_flat = pos_mask.view(B, -1)
    neighbor_flat = neighbor_mask.view(B, -1)
    far_flat = far_mask.view(B, -1)

    # 每个 batch 的正样本数量最小值，作为采样数
    num_pos = pos_flat.sum(dim=1)
    N = num_pos.clamp(min=1).min().item()

    def sample_voxels(mask_flat, num_sample):
        rand = torch.rand(mask_flat.shape, generator=g, device=device)
        rand[~mask_flat] = 2.0
        _, indices = torch.topk(-rand, k=num_sample, dim=1, largest=True)
        sampled = torch.zeros_like(mask_flat, dtype=torch.bool)
        sampled.scatter_(1, indices, True)
        return sampled

    near_sample = sample_voxels(neighbor_flat, N)
    far_sample = sample_voxels(far_flat, N)

    near_mask = near_sample.view(B, 1, D, H, W)
    far_mask = far_sample.view(B, 1, D, H, W)

    return near_mask, far_mask

def build_voxel_masks_full(target):
    """
    构建 full mask，不采样，保留 near/far 全部负样本。

    Args:
        target: (B, 1, D, H, W) occupancy GT

    Returns:
        near_mask: (B, 1, D, H, W)
        far_mask: (B, 1, D, H, W)
    """
    B, _, D, H, W = target.shape
    device = target.device

    pos_mask = target > 0
    neg_mask = target == 0

    pos_mask_f = pos_mask.float()
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    neighbor_prob = F.conv3d(pos_mask_f, kernel, padding=1)
    neighbor_mask = (neighbor_prob > 0) & neg_mask  # [B,1,D,H,W]
    far_mask = neg_mask & (~neighbor_mask)

    return neighbor_mask, far_mask

def build_voxel_weight_map(target, near_mask, far_mask, near_weight=0.2, far_weight=0.5):
    """
    构建 voxel 监督加权图。

    Args:
        target: [B, D, H, W] occupancy GT
        near_mask, far_mask: bool tensor, 来自采样器
        near_weight: float, 邻近负样本区域的权重
        far_weight: float, 远离负样本区域的权重

    Returns:
        weight_map: [B, D, H, W] float tensor
    """
    weight_map = torch.zeros_like(target, dtype=torch.float32)
    weight_map[target > 0] = 1.0
    weight_map[near_mask] = near_weight
    weight_map[far_mask] = far_weight
    return weight_map


def voxel_focus_mse(pred, target, threshold=0.1):
    """
    针对 coarse voxel 的点数预测，仅在 pred 或 target 有值的位置计算 MSE。
    
    参数：
        pred: [B, 1, D, H, W]，实数点数预测
        target: [B, 1, D, H, W]，实数点数标签
    返回：
        mse: 标量
    """
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    focus_mask = ((target > 0) | (pred > threshold)).float()
    loss_map = F.mse_loss(pred, target, reduction='none')  # [B, D, H, W]
    loss = (loss_map * focus_mask).sum() / focus_mask.sum().clamp(min=1)
    return loss

def voxel_focus_iou(pred, target, eps=1e-6):
    """
    针对 occupancy 体素的 mIoU，仅在非空区域统计。
    
    参数：
        pred: [B, 1, D, H, W]，已过 sigmoid
        target: [B, 1, D, H, W]，二值体素
    返回：
        miou: 标量
    """
    pred = pred.squeeze(1)      # [B, D, H, W]
    target = target.squeeze(1).float()

    # soft 交集
    intersection = (pred * target).sum(dim=(1, 2, 3))
    # soft 并集
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)  # [B]
    loss = 1 - iou.mean()
    return loss


def chamfer_distance(p1, p2, device):
    if p1.numel() == 0 or p2.numel() == 0:
        return torch.tensor(0.0, device=device)
    dist = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)
    return (dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()) / 2

def voxel_chamfer_loss_from_masks(pred, target, near_mask, far_mask, threshold=0.5, near_weight=0.2, far_weight=0.5):
    """
    Chamfer-style loss with separate near/far supervision.

    Args:
        pred: [B, 1, D, H, W]
        target: [B, 1, D, H, W]
        near_mask, far_mask: [B, 1, D, H, W] bool tensors
    """
    B = pred.shape[0]
    pred = pred.squeeze(1)    # [B, D, H, W]
    target = target.squeeze(1).float()

    losses = []
    for b in range(B):
        pred_bin_b = (pred[b] > threshold).float()
        target_b = target[b]
        near_b = near_mask[b, 0]  # [D, H, W]
        far_b = far_mask[b, 0]

        pos_mask = target_b > 0

        pred_coords_pos = (pos_mask & (pred_bin_b > 0)).nonzero(as_tuple=False).float()
        pos_coords = pos_mask.nonzero(as_tuple=False).float()

        pred_coords_near = (near_b & (pred_bin_b > 0)).nonzero(as_tuple=False).float()
        near_coords = near_b.nonzero(as_tuple=False).float()
        
        pred_coords_far = (far_b & (pred_bin_b > 0)).nonzero(as_tuple=False).float()
        far_coords = far_b.nonzero(as_tuple=False).float()

        loss_pos = chamfer_distance(pred_coords_pos, pos_coords, device=pred.device)
        loss_near = chamfer_distance(pred_coords_near, near_coords, device=pred.device)
        loss_far = chamfer_distance(pred_coords_far, far_coords, device=pred.device)

        loss = loss_pos + near_weight * loss_near + far_weight * loss_far
        losses.append(loss)

    return torch.stack(losses).mean()


def voxel_chamfer_loss(pred, target, threshold=0.5):
    """
    Chamfer-style loss for occupancy voxel predictions.

    参数：
        pred: [B, 1, D, W, H]，预测 occupancy，未 binarize
        target: [B, 1, D, W, H]，GT occupancy，二值
        threshold: float，阈值

    返回：
        chamfer_loss: 标量
    """
    B, _, D, W, H = pred.shape
    device = pred.device

    pred = pred.squeeze(1)
    target = target.squeeze(1).float().to(device)

    # 先binarize预测
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float()

    chamfer_losses = []

    for b in range(B):

        pred_coords = pred_bin[b].nonzero(as_tuple=False).float()  # [Np, 3]
        target_coords = target_bin[b].nonzero(as_tuple=False).float()  # [Nt, 3]

        if pred_coords.shape[0] == 0 or target_coords.shape[0] == 0:
            chamfer_losses.append(torch.tensor(0.0, device=device))
            continue

        # pred -> gt 最近距离
        dist_pred2gt = torch.cdist(pred_coords.unsqueeze(0), target_coords.unsqueeze(0)).squeeze(0)  # [Np, Nt]
        min_dist_pred2gt = dist_pred2gt.min(dim=1)[0]  # [Np]

        # gt -> pred 最近距离
        dist_gt2pred = dist_pred2gt.t()  # [Nt, Np]
        min_dist_gt2pred = dist_gt2pred.min(dim=1)[0]  # [Nt]

        loss = (min_dist_pred2gt.mean() + min_dist_gt2pred.mean()) / 2
        chamfer_losses.append(loss)

    chamfer_loss = torch.stack(chamfer_losses).mean()
    return chamfer_loss

# def focused_voxel_bce_loss(pred, target, weight_map):

#     pred = pred.squeeze(1)
#     target = target.squeeze(1).float()
#     loss = F.binary_cross_entropy(pred, target, reduction='none')
#     weighted_loss = (loss * weight_map).sum() / (weight_map > 0).sum().clamp(min=1)
#     return weighted_loss

def focused_voxel_bce_loss(pred, target, near_mask, far_mask, near_weight=0.5, far_weight=1.0):
    """
    BCE loss with separate weighting for pos, near, and far regions.

    Args:
        pred: [B, 1, D, H, W], probabilities in [0, 1]
        target: [B, 1, D, H, W], binary ground truth {0, 1}
        near_mask, far_mask: [B, 1, D, H, W] bool tensors for sampled negatives
        near_weight, far_weight: weighting factors for negative regions
    """
    pred = pred.squeeze(1).clamp(min=1e-6, max=1 - 1e-6)  # Avoid log(0)
    target = target.squeeze(1).float()
    near_mask = near_mask.squeeze(1)
    far_mask = far_mask.squeeze(1)

    loss_map = F.binary_cross_entropy(pred, target, reduction='none')

    pos_mask = target > 0

    # Ensure non-empty masks to avoid NaNs
    loss_pos = loss_map[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=pred.device)
    loss_near = loss_map[near_mask].mean() if near_mask.any() else torch.tensor(0.0, device=pred.device)
    loss_far = loss_map[far_mask].mean() if far_mask.any() else torch.tensor(0.0, device=pred.device)

    total_loss = loss_pos + near_weight * loss_near + far_weight * loss_far
    return total_loss

def focused_voxel_bce_loss_full_mask(pred, target, near_mask, far_mask):
    """
    全体参与计算，根据体素区域比例动态加权的 BCE 损失函数。
    """
    pred = pred.squeeze(1).clamp(min=1e-6, max=1 - 1e-6)
    target = target.squeeze(1).float()
    near_mask = near_mask.squeeze(1)
    far_mask = far_mask.squeeze(1)

    loss_map = F.binary_cross_entropy(pred, target, reduction='none')

    pos_mask = target > 0

    # Count voxels
    n_pos = pos_mask.sum().float()
    n_near = near_mask.sum().float()
    n_far = far_mask.sum().float()
    total = n_pos + n_near + n_far

    eps = 1e-6  # 防止除零
    inv_n_pos = 1.0 / (n_pos + eps)
    inv_n_near = 1.0 / (n_near + eps)
    inv_n_far = 1.0 / (n_far + eps)

    inv_total = inv_n_pos + inv_n_near + inv_n_far

    w_pos = inv_n_pos / inv_total
    w_near = inv_n_near / inv_total
    w_far = inv_n_far / inv_total

    # Compute average loss for each region
    loss_pos = loss_map[pos_mask].mean() if n_pos > 0 else torch.tensor(0.0, device=pred.device)
    loss_near = loss_map[near_mask].mean() if n_near > 0 else torch.tensor(0.0, device=pred.device)
    loss_far = loss_map[far_mask].mean() if n_far > 0 else torch.tensor(0.0, device=pred.device)

    total_loss = w_pos * loss_pos + w_near * loss_near + w_far * loss_far
    return total_loss

def occupied_ratio_loss(pred, target, threshold=0.5):
    """
    控制预测occupied数量接近GT数量。
    
    参数：
        pred: [B, 1, D, H, W]
        target: [B, 1, D, H, W]
    返回：
        loss: 标量
    """
    pred = pred.squeeze(1)
    target = target.squeeze(1).float().to(pred.device)

    pred_bin = (pred > threshold).float()
    # print(pred_bin.sum())
    pred_occ_ratio = pred_bin.mean(dim=(1,2,3))  # [B]
    gt_occ_ratio = (target > 0).float().mean(dim=(1,2,3))  # [B]

    loss = F.l1_loss(pred_occ_ratio, gt_occ_ratio)
    return loss

def occupied_voxel_count_loss(pred, target, threshold=0.5, mode='l1'):
    """
    比较预测和GT中occupied体素的数量差异。

    Args:
        pred: [B, 1, D, H, W] ∈ [0, 1]
        target: [B, 1, D, H, W] ∈ {0, 1}
        threshold: float, 判定pred是否为occupied
        mode: 'l1' or 'mse'

    Returns:
        loss: 标量
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float().to(pred.device)

    pred_count = pred_bin.sum()
    target_count = target_bin.sum()

    if mode == 'l1':
        return F.l1_loss(pred_count, target_count, reduction='mean')
    else:
        return F.mse_loss(pred_count, target_count, reduction='mean')
    
def occupied_voxel_count_loss_from_mask(pred, target, near_mask, far_mask,
                                        threshold=0.5,
                                        near_weight=0.1, far_weight=0.5,
                                        mode='l1'):
    """
    Compare predicted vs GT occupied voxel *ratios* in different regions.

    Args:
        pred: [B, 1, D, H, W], predicted probabilities ∈ [0,1]
        target: [B, 1, D, H, W], GT binary labels ∈ {0,1}
        near_mask, far_mask: [B, 1, D, H, W] bool tensors
        threshold: float, binarize GT occupancy
        mode: 'l1' or 'mse'
    """
    pred = pred.squeeze(1)           # [B, D, H, W]
    target = target.squeeze(1).float()
    near_mask = near_mask.squeeze(1)
    far_mask = far_mask.squeeze(1)

    device = pred.device
    loss_fn = F.l1_loss if mode == 'l1' else F.mse_loss

    # Binarize GT
    target_bin = (target > threshold).float()

    # Positive region
    pos_mask = target_bin > 0
    total_pos = pos_mask.sum().clamp(min=1)
    pred_pos_sum = pred[pos_mask].sum()
    target_pos_sum = target_bin[pos_mask].sum()
    loss_pos = loss_fn(pred_pos_sum / total_pos, target_pos_sum / total_pos)
    #print('pred_pos_sum:', pred_pos_sum, 'target_pos_sum:', target_pos_sum)

    # Near region
    total_near = near_mask.sum().clamp(min=1)
    pred_near_sum = pred[near_mask].sum()
    target_near_sum = target_bin[near_mask].sum()
    loss_near = loss_fn(pred_near_sum / total_near, target_near_sum / total_near)
    #print('pred_near_sum:', pred_near_sum, 'target_near_sum:', target_near_sum)

    # Far region
    total_far = far_mask.sum().clamp(min=1)
    pred_far_sum = pred[far_mask].sum()
    target_far_sum = target_bin[far_mask].sum()
    loss_far = loss_fn(pred_far_sum / total_far, target_far_sum / total_far)
    #print('pred_far_sum:', pred_far_sum, 'target_far_sum:', target_far_sum)

    total_loss = loss_pos + near_weight * loss_near + far_weight * loss_far
    return total_loss
    
# Main training loop function to compute loss
# def compute_loss(pred_fine, pred_coarse, target_fine, target_coarse):
    
#     # Calculate MSE loss for fine and coarse predictions
#     fine_ce_loss = focused_voxel_bce_loss(pred_fine, target_fine)
#     coarse_mse_loss = voxel_focus_mse(pred_coarse, target_coarse)

#     fine_miou_loss = voxel_focus_iou(pred_fine, target_fine)

#     return fine_ce_loss, coarse_mse_loss, fine_miou_loss

def compute_loss(pred_coarse, target_coarse):
    
    near_mask, far_mask = build_voxel_sample_masks(target_coarse)
    # bce_loss = focused_voxel_bce_loss(pred_coarse, target_coarse, near_mask, far_mask)
    near_mask_full, far_mask_full = build_voxel_masks_full(target_coarse)
    bce_loss = focused_voxel_bce_loss_full_mask(pred_coarse, target_coarse, near_mask_full, far_mask_full)
    #chamfer = voxel_chamfer_loss(pred_coarse, target_coarse)
    chamfer = torch.tensor([0.]).to(pred_coarse.device)
    #count_loss = occupied_voxel_count_loss_from_mask(pred_coarse, target_coarse, near_mask, far_mask)
    count_loss = torch.tensor([0.]).to(pred_coarse.device)
    #occ_ratio_loss = occupied_ratio_loss(pred_coarse, target_coarse)

    total_loss = bce_loss + 0.1 * chamfer + count_loss # chamfer通常系数小一些
    return total_loss, bce_loss, chamfer, count_loss



def compute_flat_mask_by_structure_tensor_batch(depth_tensor, threshold=5.0):
    """
    输入: depth_tensor: (B, 1, H, W)
    输出: flat_mask: (B, 1, H, W)，布尔张量（float类型）
    """
    device = depth_tensor.device  # 获取输入所在设备

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

    grad_x = F.conv2d(depth_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(depth_tensor, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    return (grad_mag < threshold).float()




#----------------------------------------------------------------------------------------------------------------------------------
# def gen_local_consistency_loss(mask, output_depth, target, local_lidar_depth, kernel, criterion, threshold = 3, lidar_weight = 1.0):
#     local_consistency_loss = torch.tensor(0.0, device=output_depth.device)
#     # mask 位置投影点作为中心
#     projected_mask = (mask > 0).float()
#     # 生成“相似区域”掩码：local_lidar_depth 与 target 差值小于 threshold
#     depth_diff = torch.abs(local_lidar_depth - target)
#     similarity_mask = (depth_diff < threshold).float()

#     # 和 mask 一起，提取可靠监督区域
#     match_mask = similarity_mask * projected_mask  # (B,1,H,W)

#     # 扩展为邻域搜索（3x3），只在有 match 的周围激活
#     match_neighborhood = F.conv2d(match_mask, kernel, padding=1)  # (B,1,H,W)，值在[0,9]
#     valid_region = (match_neighborhood > 0).float()  # 有邻域匹配的区域
#     valid_region = valid_region * (1 - mask)

#     # print(f"[local_consistency] valid_region nonzero count: {(valid_region > 0).sum().item()}")
#     # 使用这个区域进行 supervision（直接对 output_depth 和 target 做差）
#     if valid_region.sum() > 0:
#         supervised_pred = output_depth * valid_region
#         supervised_target = local_lidar_depth * valid_region
#         #local_consistency_loss = F.l1_loss(supervised_pred, supervised_target)
#         local_consistency_loss = criterion(supervised_pred, supervised_target)
#         lidar_weight *= mask.sum()/valid_region.sum()
#         local_consistency_loss *= lidar_weight  # 用 lidar_weight 作为权重
#     return local_consistency_loss

# def CompositeDepthLoss(output_depth, criterion, target, local_lidar_depth, mask, flat_mask, target_depth_image_5frames = None, mask_weight=1.0, smooth_weight=0.1, lidar_weight=1.0, radar_5f_weight = 1.0):
#     """
#     output_depth: 网络输出的深度图 (B, 1, H, W)
#     target: 毫米波雷达 supervision target (B, 1, H, W)
#     local_lidar_depth: 激光雷达深度图 (B, 1, H, W)
#     mask: 有效像素 mask (B, 1, H, W)
#     """
#     B, _, H, W = output_depth.shape
#     output_depth *= 255
#     ## 1. 基本的mask区域监督（加权）
#     masked_output = output_depth * mask
#     masked_target = target * mask
#     valid_pixels = mask.sum()
#     #print('in loss',valid_pixels)
#     base_loss = torch.tensor(0.0, device=output_depth.device)
#     if valid_pixels > 0:
#         base_loss = criterion(masked_output, masked_target)
#         base_loss = base_loss * mask_weight
#     else:
#         base_loss = torch.tensor(0.0, device=output_depth.device)

#     count = 0
#     threshold = 3.0  # 可调参数，local_lidar_depth 与 target 相似的阈值

#     kernel = torch.ones((1, 1, 3, 3), device=output_depth.device)
#     local_consistency_loss = gen_local_consistency_loss(mask, output_depth, target, local_lidar_depth, kernel, criterion, threshold, lidar_weight)

#     # 3. 基于深度差异生成平滑mask，判断哪些区域是深度连续的

#     #flat_mask = compute_flat_mask_by_structure_tensor_batch(local_lidar_depth, threshold=1.0).float()  # shape: (B,1,H,W)
#     smooth_loss = torch.tensor(0.0, device=output_depth.device)
#     if flat_mask is not None:
#         smooth_region_mask = flat_mask
#         # x方向差分
#         dx = torch.abs(output_depth[:, :, :, :-1] - output_depth[:, :, :, 1:])
#         dx_mask = smooth_region_mask[:, :, :, :-1]
#         dx_smooth = dx * dx_mask

#         # y方向差分
#         dy = torch.abs(output_depth[:, :, :-1, :] - output_depth[:, :, 1:, :])
#         dy_mask = smooth_region_mask[:, :, :-1, :]
#         dy_smooth = dy * dy_mask

#         # 平滑loss
#         smooth_loss = (dx_smooth.sum() + dy_smooth.sum()) / (dx_mask.sum() + dy_mask.sum() + 1e-6) * smooth_weight

#     # 4. 总loss
#     total_loss = base_loss + local_consistency_loss + smooth_loss
#     if target_depth_image_5frames is not None:
#         local_consistency_loss_radar = gen_local_consistency_loss(mask, output_depth, target, target_depth_image_5frames, kernel, criterion, threshold, radar_5f_weight)
#         total_loss += local_consistency_loss_radar
#         return total_loss,  base_loss, local_consistency_loss, smooth_loss, local_consistency_loss_radar
#     return total_loss,  base_loss, local_consistency_loss, smooth_loss, None
#----------------------------------------------------------------------------------------------------------------------------------

def CompositeDepthLoss_ori(output_depth, criterion, target, local_lidar_depth, mask, flat_mask, mask_weight=5.0, smooth_weight=1, lidar_weight=0.5):
    """
    output_depth: 网络输出的深度图 (B, 1, H, W)
    target: 毫米波雷达 supervision target (B, 1, H, W)
    local_lidar_depth: 激光雷达深度图 (B, 1, H, W)
    mask: 有效像素 mask (B, 1, H, W)
    """

    B, _, H, W = output_depth.shape

    ## 1. 基本的mask区域监督（加权）
    masked_output = output_depth * mask
    masked_target = target * mask
    valid_pixels = mask.sum()

    if valid_pixels > 0:
        base_loss = criterion(masked_output * 255, masked_target)
        base_loss = base_loss * mask_weight
    else:
        base_loss = torch.tensor(0.0, device=output_depth.device)

    local_consistency_loss = torch.tensor(0.0, device=output_depth.device)

    count = 0
    threshold = 3.0  # 可调参数，local_lidar_depth 与 target 相似的阈值

    # 卷积核用于获取局部邻域（3x3）
    kernel = torch.ones((1, 1, 3, 3), device=output_depth.device)

    # mask 位置投影点作为中心
    projected_mask = (mask > 0).float()

    # 生成“相似区域”掩码：local_lidar_depth 与 target 差值小于 threshold
    depth_diff = torch.abs(local_lidar_depth - target)
    similarity_mask = (depth_diff < threshold).float()

    # 和 mask 一起，提取可靠监督区域
    match_mask = similarity_mask * projected_mask  # (B,1,H,W)

    # 扩展为邻域搜索（3x3），只在有 match 的周围激活
    match_neighborhood = F.conv2d(match_mask, kernel, padding=1)  # (B,1,H,W)，值在[0,9]
    valid_region = (match_neighborhood > 0).float()  # 有邻域匹配的区域
    #print(f"[local_consistency] valid_region nonzero count: {(valid_region > 0).sum().item()}")
    # 使用这个区域进行 supervision（直接对 output_depth 和 target 做差）
    if valid_region.sum() > 0:
        supervised_pred = output_depth * valid_region
        supervised_target = target * valid_region
        local_consistency_loss = F.l1_loss(supervised_pred * 255, supervised_target)

        local_consistency_loss *= lidar_weight  # 用 lidar_weight 作为权重
    


    ## 3. 连续性平滑 loss
    dx = torch.abs(local_lidar_depth[:, :, :, :-1] - local_lidar_depth[:, :, :, 1:])
    dy = torch.abs(local_lidar_depth[:, :, :-1, :] - local_lidar_depth[:, :, 1:, :])

    # 3. 基于深度差异生成平滑mask，判断哪些区域是深度连续的

    #flat_mask = compute_flat_mask_by_structure_tensor_batch(local_lidar_depth, threshold=1.0).float()  # shape: (B,1,H,W)
    smooth_region_mask = flat_mask
    # x方向差分
    dx = torch.abs(output_depth[:, :, :, :-1] - output_depth[:, :, :, 1:])
    dx_mask = smooth_region_mask[:, :, :, :-1]
    dx_smooth = dx * dx_mask

    # y方向差分
    dy = torch.abs(output_depth[:, :, :-1, :] - output_depth[:, :, 1:, :])
    dy_mask = smooth_region_mask[:, :, :-1, :]
    dy_smooth = dy * dy_mask

    # 平滑loss
    smooth_loss = (dx_smooth.sum() + dy_smooth.sum()) / (dx_mask.sum() + dy_mask.sum() + 1e-6) * smooth_weight

    ## 4. 总loss
    total_loss = base_loss + local_consistency_loss + smooth_loss

    return total_loss, base_loss, local_consistency_loss, smooth_loss, None



def gen_local_consistency_loss(mask, output_depth, target, local_lidar_depth, kernel, criterion, threshold = 3, lidar_weight = 1.0):
    local_consistency_loss = torch.tensor(0.0, device=output_depth.device)
    # mask 位置投影点作为中心
    projected_mask = (mask > 0).float()
    # 生成“相似区域”掩码：local_lidar_depth 与 target 差值小于 threshold
    depth_diff = torch.abs(local_lidar_depth - target)
    similarity_mask = (depth_diff < threshold).float()

    # 和 mask 一起，提取可靠监督区域
    match_mask = similarity_mask * projected_mask  # (B,1,H,W)

    # 扩展为邻域搜索（3x3），只在有 match 的周围激活
    match_neighborhood = F.conv2d(match_mask, kernel, padding=1)  # (B,1,H,W)，值在[0,9]
    valid_region = (match_neighborhood > 0).float()  # 有邻域匹配的区域
    valid_region = valid_region * (1 - mask)
    # valid_region = valid_region * (1 - mask) * (local_lidar_depth > 0)

    # print(f"[local_consistency] valid_region nonzero count: {(valid_region > 0).sum().item()}")
    # 使用这个区域进行 supervision（直接对 output_depth 和 target 做差）
    if valid_region.sum() > 0:
        supervised_pred = output_depth * valid_region
        supervised_target = local_lidar_depth * valid_region
        #local_consistency_loss = F.l1_loss(supervised_pred, supervised_target)
        local_consistency_loss = criterion(supervised_pred, supervised_target)
        lidar_weight *= mask.sum()/valid_region.sum()
        local_consistency_loss *= lidar_weight  # 用 lidar_weight 作为权重
    return local_consistency_loss


def gen_local_consistency_loss_new(
    mask, output_depth, target, local_lidar_depth,
    kernel, criterion, threshold=3.0, lidar_weight=1.0
):
    """
    根据 local_lidar_depth 在 target 空洞区域中寻找可靠位置，与 output_depth 比较。
    
    参数:
    - mask: 原始雷达点位置 (B,1,H,W)
    - output_depth: 模型输出深度 (B,1,H,W)
    - target: 稀疏监督目标 (B,1,H,W)，非0为有监督位置
    - local_lidar_depth: 激光雷达深度图 (B,1,H,W)
    - kernel: 卷积核 (通常为 3x3 ones)
    - criterion: 损失函数，例如 L1/L2
    - threshold: 接受的误差阈值
    - lidar_weight: loss 加权因子
    """

    device = output_depth.device

    # 1. 找出 target 中无 supervision 的位置
    target_empty_mask = (target == 0).float()

    # 2. 这些空洞位置中，local_lidar_depth 不能为 0（否则也没意义）
    candidate_mask = target_empty_mask * (local_lidar_depth != 0).float()

    # 3. 统计邻域内 target ≠ 0 的个数（注意，target 非0很稀疏）
    target_nonzero = (target > 0).float()
    neighbor_count = F.conv2d(target_nonzero, kernel, padding=1)

    # 4. 求 target 邻域内非0深度的总和
    target_sum = F.conv2d(target * target_nonzero, kernel, padding=1)

    # 5. 避免除0，限定非0邻域
    avg_neighbor_depth = torch.zeros_like(target_sum)
    valid_neighbor_mask = (neighbor_count > 0)
    avg_neighbor_depth[valid_neighbor_mask] = (
        target_sum[valid_neighbor_mask] / neighbor_count[valid_neighbor_mask]
    )

    # 6. 计算 candidate 区域中，local_lidar_depth 与邻域 target 均值的误差
    depth_diff = torch.abs(avg_neighbor_depth - local_lidar_depth)

    # 7. 满足阈值条件的点：depth_diff < threshold 且 candidate_mask=1
    similarity_mask = (depth_diff < threshold).float()
    final_mask = similarity_mask * candidate_mask  # shape (B,1,H,W)

    # 8. 如果最终可用区域有点，则计算 loss
    if final_mask.sum() > 0:
        pred_depth = output_depth * final_mask
        target_depth = local_lidar_depth * final_mask
        loss = criterion(pred_depth, target_depth)

        # 按 mask 点比例调整权重
        lidar_weight *= mask.sum() / (final_mask.sum() + 1e-6)
        loss *= lidar_weight
    else:
        loss = torch.tensor(0.0, device=device)

    return loss

def CompositeDepthLoss(output_depth, criterion, target, local_lidar_depth, mask, flat_mask, target_depth_image_5frames = None, mask_weight=1.0, smooth_weight=0.1, lidar_weight=1.0, radar_5f_weight = 0.5):
    """
    output_depth: 网络输出的深度图 (B, 1, H, W)
    target: 毫米波雷达 supervision target (B, 1, H, W)
    local_lidar_depth: 激光雷达深度图 (B, 1, H, W)
    mask: 有效像素 mask (B, 1, H, W)
    """
    B, _, H, W = output_depth.shape
    output_depth *= 255

    ## 1. 基本的mask区域监督（加权）
    masked_output = output_depth * mask
    masked_target = target * mask
    valid_pixels = mask.sum()

    base_loss = torch.tensor(0.0, device=output_depth.device)
    if valid_pixels > 0:
        base_loss = criterion(masked_output, masked_target)
        base_loss = base_loss * mask_weight
    else:
        base_loss = torch.tensor(0.0, device=output_depth.device)

    count = 0
    threshold = 3.0  # 可调参数，local_lidar_depth 与 target 相似的阈值

    kernel = torch.ones((1, 1, 3, 3), device=output_depth.device)
    # local_lidar_depth[local_lidar_depth == 255] = 0.0
    #print(local_lidar_depth.min(),local_lidar_depth.max())
    #print(target.min(),target.max())

    #local_consistency_loss = torch.tensor(0.0, device=output_depth.device)
    local_consistency_loss = gen_local_consistency_loss(mask, output_depth, target, local_lidar_depth, kernel, criterion, threshold, lidar_weight)

    # 3. 基于深度差异生成平滑mask，判断哪些区域是深度连续的

    #flat_mask = compute_flat_mask_by_structure_tensor_batch(local_lidar_depth, threshold=1.0).float()  # shape: (B,1,H,W)
    smooth_loss = torch.tensor(0.0, device=output_depth.device)
    if flat_mask is not None:
        smooth_region_mask = flat_mask
        # x方向差分
        dx = torch.abs(output_depth[:, :, :, :-1] - output_depth[:, :, :, 1:])
        dx_mask = smooth_region_mask[:, :, :, :-1]
        dx_smooth = dx * dx_mask

        # y方向差分
        dy = torch.abs(output_depth[:, :, :-1, :] - output_depth[:, :, 1:, :])
        dy_mask = smooth_region_mask[:, :, :-1, :]
        dy_smooth = dy * dy_mask

        # 平滑loss
        smooth_loss = (dx_smooth.sum() + dy_smooth.sum()) / (dx_mask.sum() + dy_mask.sum() + 1e-6) * smooth_weight

    # 4. 总loss
    total_loss = base_loss + local_consistency_loss + smooth_loss
    if target_depth_image_5frames is not None:
        mask_5frames = (target_depth_image_5frames > 0).float()
        base_loss_5frames = torch.tensor(0.0, device=output_depth.device)
        masked_output = output_depth * mask_5frames
        masked_target = target_depth_image_5frames * mask_5frames
        valid_pixels = mask_5frames.sum()
        if valid_pixels > 0:
            base_loss_5frames = criterion(masked_output, masked_target)
            base_loss_5frames = base_loss_5frames * radar_5f_weight
        else:
            base_loss_5frames = torch.tensor(0.0, device=output_depth.device)
        total_loss += base_loss_5frames
        return total_loss,  base_loss, local_consistency_loss, smooth_loss, base_loss_5frames
    return total_loss,  base_loss, local_consistency_loss, smooth_loss, None



#def CompositeDepthLoss_old(output_depth, criterion, target, local_lidar_depth, mask, flat_mask, target_depth_image_5frames = None, mask_weight=1.0, smooth_weight=0.1, lidar_weight=1.0):
def CompositeDepthLoss_old(output_depth, criterion, target, local_lidar_depth, mask, flat_mask, mask_weight=1.0, smooth_weight=0.1, lidar_weight=1.0):
    """
    output_depth: 网络输出的深度图 (B, 1, H, W)
    target: 毫米波雷达 supervision target (B, 1, H, W)
    local_lidar_depth: 激光雷达深度图 (B, 1, H, W)
    mask: 有效像素 mask (B, 1, H, W)
    """

    B, _, H, W = output_depth.shape
    output_depth *= 255
    ## 1. 基本的mask区域监督（加权）
    masked_output = output_depth * mask
    masked_target = target * mask
    valid_pixels = mask.sum()

    if valid_pixels > 0:
        base_loss = criterion(masked_output, masked_target)
        base_loss = base_loss * mask_weight
    else:
        base_loss = torch.tensor(0.0, device=output_depth.device)

    local_consistency_loss = torch.tensor(0.0, device=output_depth.device)

    count = 0
    threshold = 3.0  # 可调参数，local_lidar_depth 与 target 相似的阈值

    # 卷积核用于获取局部邻域（3x3）
    kernel = torch.ones((1, 1, 3, 3), device=output_depth.device)

    # mask 位置投影点作为中心
    projected_mask = (mask > 0).float()

    # 生成“相似区域”掩码：local_lidar_depth 与 target 差值小于 threshold
    depth_diff = torch.abs(local_lidar_depth - target)
    similarity_mask = (depth_diff < threshold).float()

    # 和 mask 一起，提取可靠监督区域
    match_mask = similarity_mask * projected_mask  # (B,1,H,W)

    # 扩展为邻域搜索（3x3），只在有 match 的周围激活
    match_neighborhood = F.conv2d(match_mask, kernel, padding=1)  # (B,1,H,W)，值在[0,9]
    valid_region = (match_neighborhood > 0).float()  # 有邻域匹配的区域
    valid_region = valid_region * (1 - mask)
    # print(f"[local_consistency] valid_region nonzero count: {(valid_region > 0).sum().item()}")
    # 使用这个区域进行 supervision（直接对 output_depth 和 target 做差）
    if valid_region.sum() > 0:
        supervised_pred = output_depth * valid_region
        supervised_target = local_lidar_depth * valid_region
        #local_consistency_loss = F.l1_loss(supervised_pred, supervised_target)
        local_consistency_loss = criterion(supervised_pred, supervised_target)
        local_consistency_loss *= lidar_weight  # 用 lidar_weight 作为权重
    



    # 3. 基于深度差异生成平滑mask，判断哪些区域是深度连续的

    #flat_mask = compute_flat_mask_by_structure_tensor_batch(local_lidar_depth, threshold=1.0).float()  # shape: (B,1,H,W)
    smooth_region_mask = flat_mask
    # x方向差分
    dx = torch.abs(output_depth[:, :, :, :-1] - output_depth[:, :, :, 1:])
    dx_mask = smooth_region_mask[:, :, :, :-1]
    dx_smooth = dx * dx_mask

    # y方向差分
    dy = torch.abs(output_depth[:, :, :-1, :] - output_depth[:, :, 1:, :])
    dy_mask = smooth_region_mask[:, :, :-1, :]
    dy_smooth = dy * dy_mask

    # 平滑loss
    smooth_loss = (dx_smooth.sum() + dy_smooth.sum()) / (dx_mask.sum() + dy_mask.sum() + 1e-6) * smooth_weight

    ## 4. 总loss
    total_loss = base_loss + local_consistency_loss + smooth_loss

    return total_loss, base_loss, local_consistency_loss, smooth_loss, None