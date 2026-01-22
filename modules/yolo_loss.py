import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    """YOLOv3 Loss Function with WIoU v3 Support"""

    def __init__(self, num_classes, anchors, img_size=640, use_wiou=True):
        """
        Args:
            num_classes: 类别数量
            anchors: 每个尺度的anchor boxes [[w1,h1,w2,h2,...], [...], [...]]
            img_size: 输入图像尺寸
            use_wiou: 是否使用 WIoU v3 计算 box loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_wiou = use_wiou

        # 将anchors转换为tensor并归一化
        self.anchors = []
        for scale_anchors in anchors:
            anchors_pairs = []
            for i in range(0, len(scale_anchors), 2):
                anchors_pairs.append([scale_anchors[i], scale_anchors[i+1]])
            self.anchors.append(torch.tensor(anchors_pairs, dtype=torch.float32))

        # 使用sum reduction来更稳定
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # 调整loss权重，减小初始loss
        self.lambda_coord = 1.0
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_cls = 1.0

        # WIoU v3 相关参数
        self.register_buffer('self_iou_mean', torch.tensor(0.0))
        self.iou_momentum = 0.9

    def forward(self, predictions, targets):
        """
        计算loss
        Args:
            predictions: 模型输出列表
            targets: ground truth [num_boxes, 6] - (batch_idx, class_id, x, y, w, h)
        Returns:
            total_loss: 总损失
        """
        device = predictions[0].device

        # 空目标处理
        if targets.shape[0] == 0:
            return self._compute_empty_target_loss(predictions)

        # 初始化
        loss_box, loss_obj, loss_cls = 0.0, 0.0, 0.0
        num_targets, num_total_pixels = 0, 0

        # 对每个尺度计算loss
        for scale_idx, pred in enumerate(predictions):
            batch_size, num_anchors, grid_size = pred.shape[0], len(self.anchors[scale_idx]), pred.shape[2]
            num_total_pixels += batch_size * num_anchors * grid_size * grid_size

            # 重塑预测并提取各部分
            pred_xy, pred_wh, pred_conf, pred_cls = self._reshape_prediction(pred, num_anchors, grid_size)

            # 构建目标张量
            target_obj, target_box, target_cls_scale, n_targets = self._build_targets(
                targets, batch_size, num_anchors, grid_size, scale_idx, device
            )
            num_targets += n_targets

            # 计算mask
            obj_mask = target_obj.squeeze(-1) > 0
            noobj_mask = target_obj.squeeze(-1) == 0

            # Box loss
            if obj_mask.sum() > 0:
                if self.use_wiou:
                    loss_box += self._compute_wiou_loss(
                        pred_xy, pred_wh, target_box, obj_mask, scale_idx, grid_size, device
                    )
                else:
                    loss_box += self._compute_box_loss_mse(pred_xy, pred_wh, target_box, obj_mask)

            # Objectness loss
            loss_obj += self._compute_obj_loss(pred_conf, target_obj, obj_mask, noobj_mask)

            # Class loss
            if obj_mask.sum() > 0:
                loss_cls += self.bce_loss(pred_cls[obj_mask], target_cls_scale[obj_mask])

        # 归一化并计算总loss
        return self._compute_total_loss(loss_box, loss_obj, loss_cls, num_targets, num_total_pixels, device)

    # ==================== 辅助方法 ====================

    def _compute_empty_target_loss(self, predictions):
        """计算空目标时的loss（仅noobj loss）"""
        loss_obj = torch.tensor(0.0, device=predictions[0].device)
        for scale_idx, pred in enumerate(predictions):
            # pred shape: (bs, na, ny, nx, no) - Detect 层训练模式输出
            pred_conf = pred[..., 4:5]
            loss_obj += self.lambda_noobj * self.bce_loss(pred_conf, torch.zeros_like(pred_conf))

        total_pixels = sum([p.shape[0] * p.shape[1] * p.shape[2] * p.shape[3]
                           for i, p in enumerate(predictions)])
        return loss_obj / max(total_pixels, 1)

    def _reshape_prediction(self, pred, num_anchors, grid_size):
        """重塑预测并提取各部分

        输入格式：(bs, na, ny, nx, no) - Detect 层训练模式输出
        """
        return (
            torch.sigmoid(pred[..., 0:2]),  # xy
            pred[..., 2:4],                  # wh
            pred[..., 4:5],                  # conf
            pred[..., 5:]                    # cls
        )

    def _build_targets(self, targets, batch_size, num_anchors, grid_size, scale_idx, device):
        """为每个尺度构建目标张量并分配anchor"""
        target_obj = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
        target_box = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 4, device=device)
        target_cls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, self.num_classes, device=device)
        num_targets = 0

        for target in targets:
            if target.sum() == 0:
                continue

            batch_idx, class_id = int(target[0]), int(target[1])
            x, y, w, h = target[2:6]

            if batch_idx >= batch_size or class_id >= self.num_classes:
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue

            # 转换到grid尺度
            gx, gy, gw, gh = x * grid_size, y * grid_size, w * grid_size, h * grid_size
            gi, gj = int(min(gx, grid_size - 1)), int(min(gy, grid_size - 1))

            # 选择最佳anchor
            gt_box = torch.tensor([0, 0, gw, gh], device=device)
            anchor_boxes = torch.cat([
                torch.zeros(num_anchors, 2, device=device),
                self.anchors[scale_idx].to(device) / (self.img_size / grid_size)
            ], dim=1)
            best_anchor = torch.argmax(self.bbox_iou(gt_box.unsqueeze(0), anchor_boxes))

            # 设置目标
            target_obj[batch_idx, best_anchor, gj, gi, 0] = 1.0
            target_box[batch_idx, best_anchor, gj, gi] = torch.tensor([
                gx - gi, gy - gj,
                torch.log(gw / (self.anchors[scale_idx][best_anchor, 0] / (self.img_size / grid_size) + 1e-16) + 1e-16),
                torch.log(gh / (self.anchors[scale_idx][best_anchor, 1] / (self.img_size / grid_size) + 1e-16) + 1e-16)
            ], device=device)
            target_cls[batch_idx, best_anchor, gj, gi, class_id] = 1.0
            num_targets += 1

        return target_obj, target_box, target_cls, num_targets

    def _compute_wiou_loss(self, pred_xy, pred_wh, target_box, obj_mask, scale_idx, grid_size, device):
        """计算 WIoU v3 loss"""
        # 提取预测框
        if obj_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pred_xy_masked, pred_wh_masked = pred_xy[obj_mask], pred_wh[obj_mask]
        mask_indices = torch.where(obj_mask)
        _, anchor_idxs, gy_idxs, gx_idxs = mask_indices

        pred_boxes = torch.stack([
            pred_xy_masked[:, 0] + gx_idxs.float(),
            pred_xy_masked[:, 1] + gy_idxs.float(),
            torch.exp(pred_wh_masked[:, 0]),
            torch.exp(pred_wh_masked[:, 1])
        ], dim=1)

        # 提取真实框
        target_box_masked = target_box[obj_mask]
        anchor_dims = torch.stack([
            self.anchors[scale_idx][anchor_idxs, 0] / (self.img_size / grid_size),
            self.anchors[scale_idx][anchor_idxs, 1] / (self.img_size / grid_size)
        ], dim=1)

        target_boxes = torch.stack([
            target_box_masked[:, 0] + gx_idxs.float(),
            target_box_masked[:, 1] + gy_idxs.float(),
            torch.exp(target_box_masked[:, 2]) * anchor_dims[:, 0],
            torch.exp(target_box_masked[:, 3]) * anchor_dims[:, 1]
        ], dim=1)

        # 更新滑动平均
        with torch.no_grad():
            normal_iou = torch.diag(self.bbox_iou(target_boxes, pred_boxes)).mean()
            batch_l_iou = 1.0 - normal_iou
            if self.self_iou_mean == 0:
                self.self_iou_mean = batch_l_iou
            else:
                self.self_iou_mean = self.iou_momentum * self.self_iou_mean + (1 - self.iou_momentum) * batch_l_iou

        # 计算 WIoU v3 loss
        wious = torch.diag(self.bbox_iou(target_boxes, pred_boxes, WIoU=True, self_iou_mean=float(self.self_iou_mean)))
        return (1.0 - wious).sum()

    def _compute_box_loss_mse(self, pred_xy, pred_wh, target_box, obj_mask):
        """计算 MSE box loss"""
        return (
            self.mse_loss(pred_xy[obj_mask], target_box[..., :2][obj_mask]) +
            self.mse_loss(pred_wh[obj_mask], target_box[..., 2:][obj_mask])
        )

    def _compute_obj_loss(self, pred_conf, target_obj, obj_mask, noobj_mask):
        """计算 objectness loss"""
        loss = torch.tensor(0.0, device=pred_conf.device)
        if obj_mask.sum() > 0:
            loss += self.bce_loss(pred_conf[obj_mask], target_obj[obj_mask])
        if noobj_mask.sum() > 0:
            loss += self.lambda_noobj * self.bce_loss(pred_conf[noobj_mask], target_obj[noobj_mask])
        return loss

    def _compute_total_loss(self, loss_box, loss_obj, loss_cls, num_targets, num_total_pixels, device):
        """计算总loss并进行归一化"""
        num_targets = max(num_targets, 1)
        loss_box = loss_box / num_targets
        loss_obj = loss_obj / num_total_pixels
        loss_cls = loss_cls / num_targets

        total_loss = (
            self.lambda_coord * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )

        if torch.isnan(total_loss):
            print(f"Warning: NaN detected! box={loss_box:.4f}, obj={loss_obj:.4f}, cls={loss_cls:.4f}")
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss

    # ==================== IoU 计算 ====================

    def bbox_iou(self, box1, box2, GIoU=False, DIoU=False, CIoU=False, WIoU=False, self_iou_mean=None, eps=1e-7):
        """
        计算两个box的IoU及其变体
        Args:
            box1: [N, 4] (x, y, w, h) 中心坐标格式
            box2: [M, 4] (x, y, w, h) 中心坐标格式
            WIoU: Wise IoU
            self_iou_mean: WIoU v3 的滑动平均 L_iou 值
        Returns:
            [N, M] IoU矩阵
        """
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

        # Intersection & Union
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
               (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        if not (CIoU or DIoU or GIoU or WIoU):
            return iou

        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

        if CIoU or DIoU or WIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = (x2 - x1).pow(2) + (y2 - y1).pow(2)

            if WIoU:
                dist_term = torch.exp(rho2 / c2.detach() * 3)
                if self_iou_mean is None:
                    return iou * dist_term  # WIoU v1
                # WIoU v3
                l_iou = 1.0 - iou
                beta = l_iou.detach() / self_iou_mean
                alpha, delta = 1.9, 3
                r = beta / (delta * (alpha ** (beta - delta)))
                return iou - (1 - dist_term) * r

            if CIoU:
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2  # DIoU

        # GIoU
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area


# ==================== Standalone IoU function for anchor-free ====================

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, WIoU=False, self_iou_mean=None, eps=1e-7):
    """Calculate IoU between boxes (standalone function for anchor-free)

    Args:
        box1: [N, 4] predictions
        box2: [M, 4] targets
        xywh: if True, input is (x, y, w, h); if False, input is (x1, y1, x2, y2)
        GIoU: compute Generalized IoU
        DIoU: compute Distance IoU
        CIoU: compute Complete IoU
        WIoU: compute Wise IoU
        self_iou_mean: WIoU v3's moving average L_iou value
        eps: small value for numerical stability

    Returns:
        [N, M] IoU matrix
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center distance**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner for anchor-free detection

    Assigns targets based on alignment between classification and localization.
    Follows the TAL (Task-Aligned Learning) principle from TOOD.
    """

    def __init__(self, topk=64, num_classes=80, alpha=0.5, beta=2.0):
        """
        Args:
            topk: number of top candidates to select per GT
            num_classes: number of classes
            alpha: weight for alignment score (classification)
            beta: exponent for alignment score (localization)
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """Compute aligned assignment

        Args:
            pd_scores: (bs, n_points, nc) predicted classification scores
            pd_bboxes: (bs, n_points, 4) predicted bboxes in xyxy format (pixels)
            anchor_points: (n_points, 2) anchor point coordinates (pixels)
            gt_labels: (bs, n_gt, 1) ground truth labels
            gt_bboxes: (bs, n_gt, 4) ground truth bboxes in xyxy format (pixels)
            mask_gt: (bs, n_gt, 1) mask for valid GT boxes

        Returns:
            target_labels: (bs, n_points) target labels
            target_bboxes: (bs, n_points, 4) target bboxes in xyxy format
            target_scores: (bs, n_points, nc) target scores (soft labels)
            fg_mask: (bs, n_points) foreground mask
            target_gt_idx: (bs, n_points) assigned GT index
        """
        bs, n_points = pd_scores.shape[:2]
        device = pd_scores.device

        # Get valid GT boxes
        mask_gt = mask_gt.squeeze(-1)  # (bs, n_gt)
        gt_labels = gt_labels.squeeze(-1)  # (bs, n_gt)

        # Compute alignment metrics
        align_metrics = self._compute_alignment_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt
        )  # (bs, n_gt, n_points)

        # Select top-k candidates for each GT
        mask_pos, target_gt_idx = self._select_topk_candidates(
            align_metrics, mask_gt
        )  # mask_pos: (bs, n_gt, n_points)

        # Assign targets
        target_labels, target_bboxes, target_scores = self._assign_targets(
            pd_scores, gt_labels, gt_bboxes, mask_pos, target_gt_idx, mask_gt
        )

        # Foreground mask
        fg_mask = mask_pos.sum(dim=1) > 0  # (bs, n_points)

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    def _compute_alignment_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment scores between predictions and GT

        Alignment = (cls_score^alpha) * (IoU^beta)
        """
        bs, n_points, nc = pd_scores.shape
        n_gt = gt_labels.shape[1]
        device = pd_scores.device

        pd_scores = pd_scores.sigmoid()  # (bs, n_points, nc)

        # Create alignment matrix
        align_metrics = torch.zeros(bs, n_gt, n_points, device=device)

        for b in range(bs):
            valid_gt = mask_gt[b]  # (n_gt,)
            if not valid_gt.any():
                continue

            valid_gt_idx = valid_gt.nonzero(as_tuple=True)[0]
            for gt_idx in valid_gt_idx:
                gt_cls = gt_labels[b, gt_idx].item()  # class index

                # Classification score for this class: (n_points,)
                cls_score = pd_scores[b, :, gt_cls]

                # Compute IoU between all predictions and this GT bbox
                # pd_bboxes[b]: (n_points, 4), gt_bboxes[b, gt_idx]: (4,)
                # Need to expand gt_bbox to (n_points, 4) for pairwise computation
                gt_bbox_expanded = gt_bboxes[b:b+1, gt_idx:gt_idx+1, :].expand(-1, n_points, -1)  # (1, n_points, 4)
                gt_bbox_flat = gt_bbox_expanded.reshape(-1, 4)  # (n_points, 4)
                pd_bbox_flat = pd_bboxes[b]  # (n_points, 4)

                # Compute IoU: returns (n_points, n_points) but we want diagonal
                iou_matrix = bbox_iou(pd_bbox_flat, gt_bbox_flat, xywh=False, CIoU=False)  # (n_points, n_points)
                iou_score = iou_matrix.diag()  # (n_points,)

                # Alignment score: (cls_score^alpha) * (IoU^beta)
                # Note: We don't filter by IoU > 0 because initial predictions may have no overlap
                # The top-k selection will naturally pick the best candidates
                # Add epsilon before exponentiation to avoid zero values
                alignment = ((cls_score + 1e-5) ** self.alpha) * ((iou_score.abs() + 1e-5) ** self.beta)
                align_metrics[b, gt_idx, :] = alignment

        return align_metrics

    def _select_topk_candidates(self, align_metrics, mask_gt):
        """Select top-k candidates for each GT"""
        bs, n_gt, n_points = align_metrics.shape
        device = align_metrics.device

        # Get top-k indices
        topk_metrics, topk_idxs = torch.topk(align_metrics, self.topk, dim=-1)
        # topk_idxs: (bs, n_gt, topk)

        # Create mask
        mask_pos = torch.zeros(bs, n_gt, n_points, dtype=torch.bool, device=device)
        target_gt_idx = torch.zeros(bs, n_points, dtype=torch.long, device=device)

        for b in range(bs):
            valid_gt = mask_gt[b]
            if not valid_gt.any():
                continue

            valid_gt_idx = valid_gt.nonzero(as_tuple=True)[0]
            for gt_i, gt_idx in enumerate(valid_gt_idx):
                idxs = topk_idxs[b, gt_i, :]  # (topk,)
                mask_pos[b, gt_idx, idxs] = True
                target_gt_idx[b, idxs] = gt_idx

        # Additional filtering: ensure selected points have positive overlap
        # This prevents selecting points that are far outside the bbox
        # We use the overlaps tensor from the parent context to filter

        return mask_pos, target_gt_idx

    def _assign_targets(self, pd_scores, gt_labels, gt_bboxes, mask_pos, target_gt_idx, mask_gt):
        """Assign targets based on positive positions"""
        bs, n_points, nc = pd_scores.shape
        device = pd_scores.device

        target_labels = torch.zeros(bs, n_points, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(bs, n_points, 4, device=device)
        target_scores = torch.zeros(bs, n_points, nc, device=device)

        for b in range(bs):
            valid_gt = mask_gt[b]
            if not valid_gt.any():
                continue

            valid_gt_idx = valid_gt.nonzero(as_tuple=True)[0]
            for gt_i, gt_idx in enumerate(valid_gt_idx):
                mask = mask_pos[b, gt_idx]  # (n_points,)
                if mask.sum() == 0:
                    continue

                # Assign labels
                target_labels[b, mask] = gt_labels[b, gt_idx]

                # Assign bboxes
                target_bboxes[b, mask] = gt_bboxes[b, gt_idx]

                # Assign scores (soft label based on overlap)
                cls_idx = gt_labels[b, gt_idx]
                target_scores[b, mask, cls_idx] = 1.0

        return target_labels, target_bboxes, target_scores


class BboxLoss(nn.Module):
    """Bounding box loss with IoU and DFL"""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask):
        """Compute bbox loss

        Args:
            pred_dist: (bs, n_points, 4 * reg_max) DFL predictions
            pred_bboxes: (bs, n_points, 4) predicted bboxes in xyxy format
            anchor_points: (n_points, 2) anchor points (pixels)
            target_bboxes: (bs, n_points, 4) target bboxes in xyxy format (normalized by stride)
            target_scores: (bs, n_points, nc) target scores
            target_scores_sum: scalar, sum of target scores for normalization
            fg_mask: (bs, n_points) foreground mask

        Returns:
            loss_iou: IoU loss
            loss_dfl: DFL loss
        """
        device = pred_bboxes.device
        dtype = pred_bboxes.dtype

        if fg_mask.sum() == 0:
            # No foreground samples, return zero with gradient graph
            # Use a single element to maintain computational graph
            dummy = pred_bboxes[0, 0, 0:1].detach() * 0
            return dummy, dummy

        # Weight by target scores
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # (n_fg, 1)

        # IoU loss - use simple IoU for better gradient flow
        # CIoU can have issues when boxes don't overlap during early training
        iou = bbox_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False, CIoU=False, GIoU=False
        )
        # Ensure IoU is non-negative for stable gradients
        iou = iou.clamp(min=0)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.zeros((), device=device, dtype=dtype)

        return loss_iou, loss_dfl


class DFLoss(nn.Module):
    """Distribution Focal Loss"""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        """Compute DFL loss

        Args:
            pred_dist: (n * 4, reg_max) predicted distance distributions (flattened)
            target: (n, 4) target distances in ltrb format (normalized to [0, reg_max-1])

        Returns:
            loss: (n, 1) DFL loss averaged over 4 boundaries
        """
        # Clamp target to valid range
        target = target.clamp(0, self.reg_max - 1 - 0.01)

        # Target left and right bin indices
        target_left = target.long()
        target_right = target_left + 1

        # Weights for linear interpolation
        weight_left = (target_right - target)
        weight_right = 1 - weight_left

        # Cross-entropy loss
        # Following ultralytics: reshape pred_dist to (n*4, reg_max), target to (n*4,)
        loss_left = F.cross_entropy(pred_dist, target_left.view(-1), reduction='none').view(target.shape)
        loss_right = F.cross_entropy(pred_dist, target_right.view(-1), reduction='none').view(target.shape)

        # Interpolate between left and right
        loss = loss_left * weight_left + loss_right * weight_right  # (n, 4)

        # Average over 4 boundaries, following ultralytics
        return loss.mean(-1, keepdim=True)  # (n, 1)


class YOLOLossAnchorFree(nn.Module):
    """Anchor-free YOLO Loss with DFL (Distribution Focal Loss)

    Modern anchor-free detection loss used in YOLOv8/v11:
    - No anchor box assignment
    - Decoupled classification and regression
    - DFL for bbox distribution prediction
    - Task-aligned target assignment
    """

    def __init__(self, num_classes, reg_max=16, img_size=640, use_dfl=True):
        """
        Args:
            num_classes: number of classes
            reg_max: number of distribution bins for DFL
            img_size: input image size
            use_dfl: whether to use DFL loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        # Loss weights (adjusted for better convergence)
        # Reduced box weight from 7.5 to 2.0 for better balance with cls and dfl
        self.hyp_box = 2.0   # box gain (reduced for better convergence)
        self.hyp_cls = 0.5   # cls gain
        self.hyp_dfl = 1.5   # dfl gain

        # BCE loss for classification (reduction='none' for manual weighting)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # DFL projection bins
        self.register_buffer(
            'proj',
            torch.arange(reg_max, dtype=torch.float32)
        )

        # Strides for each scale
        self.stride = torch.tensor([8., 16., 32.])

        # Task-aligned assigner
        self.assigner = TaskAlignedAssigner(topk=64, num_classes=num_classes, alpha=0.5, beta=2.0)

        # Bbox loss (IoU + DFL)
        self.bbox_loss = BboxLoss(reg_max) if use_dfl else None

    def forward(self, predictions, targets):
        """
        Compute anchor-free loss following ultralytics design

        Args:
            predictions: dict with 'cls' and 'reg' keys
                - cls: list of (bs, nc, h, w) for each scale
                - reg: list of (bs, 4, reg_max, h, w) for each scale
            targets: ground truth [num_boxes, 6] - (batch_idx, class_id, x, y, w, h)
                    normalized coordinates [0, 1]

        Returns:
            total_loss: dict with individual losses
        """
        device = predictions['cls'][0].device
        dtype = predictions['cls'][0].dtype

        # Empty targets handling
        if targets.shape[0] == 0:
            return self._compute_empty_target_loss(predictions)

        batch_size = predictions['cls'][0].shape[0]

        # Flatten predictions from all scales
        cls_preds, reg_preds, anchor_points, stride_tensor = self._make_anchor_points(predictions)
        # cls_preds: (bs, total_points, nc)
        # reg_preds: (bs, total_points, 4 * reg_max)
        # anchor_points: (total_points, 2)
        # stride_tensor: (total_points, 2)

        # Decode predictions to bboxes (xyxy format)
        # _bbox_decode returns bboxes in GRID coordinates
        pred_bboxes_grid = self._bbox_decode(anchor_points, reg_preds)  # (bs, total_points, 4)

        # Convert grid coordinates to pixel coordinates for assigner
        # stride_tensor: (total_points, 2) -> need to expand to match pred_bboxes
        stride_expanded = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)  # (bs, total_points, 2)
        stride_full = torch.cat([stride_expanded, stride_expanded], dim=-1)  # (bs, total_points, 4)
        pred_bboxes_pixels = pred_bboxes_grid * stride_full  # (bs, total_points, 4)

        # Prepare targets
        gt_bboxes, gt_labels, batch_idx, mask_gt = self._preprocess_targets(
            targets, batch_size
        )  # gt_bboxes: (bs, max_gt, 4) in xyxy (pixels), gt_labels: (bs, max_gt, 1)

        # Task-aligned assignment
        # Both pred_bboxes_pixels and gt_bboxes are now in pixel coordinates
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            cls_preds.detach().sigmoid(),
            pred_bboxes_pixels.detach(),
            anchor_points * stride_tensor,  # anchor points in pixels
            gt_labels,
            gt_bboxes,
            mask_gt
        )

        # Compute target scores sum for normalization
        # IMPORTANT: Use torch.clamp to ensure numerical stability and correct device placement
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        # === Classification Loss ===
        # Following ultralytics: compute BCE on ALL positions, then divide by target_scores_sum
        # This is crucial - negative positions contribute to loss too!
        bce_loss = self.bce(cls_preds, target_scores.to(dtype))
        loss_cls = bce_loss.sum() / target_scores_sum

        # Add numerical stability check to prevent gradient explosion
        # If cls_loss is abnormally high, clip it
        loss_cls = torch.clamp(loss_cls, max=50.0)

        # === Bbox Loss (IoU + DFL) ===
        loss_box = torch.zeros((), device=device, dtype=dtype)
        loss_dfl = torch.zeros((), device=device, dtype=dtype)

        if fg_mask.sum() > 0 and self.bbox_loss is not None:
            # Normalize target bboxes by stride for loss computation
            # target_bboxes: (bs, n_points, 4) in xyxy format
            # stride_tensor: (n_points, 2) representing (stride_x, stride_y)
            stride_expanded = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)  # (bs, n_points, 2)
            # Divide x coordinates by stride_x and y coordinates by stride_y
            # Stack stride to match bbox dimensions: (bs, n_points, 4)
            stride_full = torch.cat([stride_expanded, stride_expanded], dim=-1)  # (bs, n_points, 4)
            target_bboxes_norm = target_bboxes / stride_full  # (bs, n_points, 4)
            loss_box, loss_dfl = self.bbox_loss(
                reg_preds,
                pred_bboxes_grid,  # Grid coordinates, same as target_bboxes_norm
                anchor_points,
                target_bboxes_norm,
                target_scores,
                target_scores_sum,
                fg_mask
            )

        # Apply gain weights
        total_box = loss_box * self.hyp_box
        total_cls = loss_cls * self.hyp_cls
        total_dfl = loss_dfl * self.hyp_dfl

        # Stack losses for ultralytics-style return
        # loss tensor: [box, cls, dfl] (already multiplied by gains)
        loss = torch.stack([total_box, total_cls, total_dfl])

        # Return (loss * batch_size for backward, loss.detach() for logging)
        # Following ultralytics format: the second value is NOT multiplied by batch_size
        return loss * batch_size, loss.detach()

    def _make_anchor_points(self, predictions):
        """Create anchor points and flatten predictions from all scales

        Returns:
            cls_preds: (bs, total_points, nc)
            reg_preds: (bs, total_points, 4 * reg_max)
            anchor_points: (total_points, 2)
            stride_tensor: (total_points, 2)
        """
        device = predictions['cls'][0].device
        cls_list, reg_list, anchor_list, stride_list = [], [], [], []

        for scale_idx, (cls_pred, reg_pred) in enumerate(zip(predictions['cls'], predictions['reg'])):
            bs, nc, ny, nx = cls_pred.shape
            stride = self.stride[scale_idx].item()

            # Generate anchor points
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=device), torch.arange(nx, device=device)],
                indexing='ij'
            )
            anchor_points = torch.stack([xv, yv], dim=-1).float()  # (ny, nx, 2)
            anchor_points = anchor_points.view(-1, 2)  # (ny*nx, 2)

            # Stride tensor
            stride_tensor = torch.full((anchor_points.shape[0], 2), stride, device=device)

            # Flatten predictions
            cls_flat = cls_pred.view(bs, nc, -1).permute(0, 2, 1)  # (bs, ny*nx, nc)
            reg_flat = reg_pred.view(bs, 4, self.reg_max, -1)  # (bs, 4, reg_max, ny*nx)
            reg_flat = reg_flat.permute(0, 3, 1, 2).contiguous()  # (bs, ny*nx, 4, reg_max)
            reg_flat = reg_flat.view(bs, -1, 4 * self.reg_max)  # (bs, ny*nx, 4*reg_max)

            cls_list.append(cls_flat)
            reg_list.append(reg_flat)
            anchor_list.append(anchor_points)
            stride_list.append(stride_tensor)

        # Concatenate all scales
        cls_preds = torch.cat(cls_list, dim=1)  # (bs, total_points, nc)
        reg_preds = torch.cat(reg_list, dim=1)  # (bs, total_points, 4*reg_max)
        anchor_points = torch.cat(anchor_list, dim=0)  # (total_points, 2)
        stride_tensor = torch.cat(stride_list, dim=0)  # (total_points, 2)

        return cls_preds, reg_preds, anchor_points, stride_tensor

    def _bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted distributions to bboxes

        Following ultralytics: ltrb distances are in grid coordinates, not pixels.
        This allows consistent IoU computation across all scales.

        Args:
            anchor_points: (n_points, 2) anchor point coordinates (grid indices)
            pred_dist: (bs, n_points, 4 * reg_max) predicted distributions

        Returns:
            bboxes: (bs, n_points, 4) bboxes in xyxy format (grid coordinates)
        """
        bs, n_points = pred_dist.shape[:2]
        device = pred_dist.device

        # Reshape and apply softmax
        pred_dist = pred_dist.view(bs, n_points, 4, self.reg_max)  # (bs, n_points, 4, reg_max)
        pred_dist = pred_dist.softmax(3)  # (bs, n_points, 4, reg_max)

        # Project to distances using softmax * [0, 1, ..., reg_max-1]
        # Result is in range [0, reg_max-1] = [0, 15]
        proj = self.proj.to(device).view(1, 1, 1, -1)  # (1, 1, 1, reg_max)
        ltrb = (pred_dist * proj).sum(3)  # (bs, n_points, 4)

        # Convert ltrb to xyxy in grid coordinate space
        # anchor_points are in grid units, ltrb are in grid units
        lt, rb = ltrb.chunk(2, dim=-1)  # Each: (bs, n_points, 2)

        # Expand anchor_points to match batch dimension
        # anchor_points: (n_points, 2) -> (1, n_points, 2)
        anchor = anchor_points.unsqueeze(0)  # (1, n_points, 2)

        x1y1 = anchor - lt  # (bs, n_points, 2)
        x2y2 = anchor + rb  # (bs, n_points, 2)

        bboxes = torch.cat([x1y1, x2y2], dim=-1)  # (bs, n_points, 4)

        # Clip bboxes to valid range [0, max_grid_size]
        # This prevents negative coordinates and ensures valid IoU computation
        # The max grid size is 80 for P3/8, 40 for P4/16, 20 for P5/32
        # But since we concatenate all scales, we use 80 as the max
        bboxes = bboxes.clamp(min=0, max=80)

        return bboxes

    def _preprocess_targets(self, targets, batch_size):
        """Preprocess targets into batch format

        Args:
            targets: (N, 6) - (batch_idx, class_id, x, y, w, h) normalized [0, 1]
            batch_size: batch size

        Returns:
            gt_bboxes: (bs, max_gt, 4) in xyxy format (pixels)
            gt_labels: (bs, max_gt, 1)
            batch_idx: (bs, max_gt, 1)
            mask_gt: (bs, max_gt, 1)
        """
        device = targets.device

        if targets.shape[0] == 0:
            max_gt = 0
        else:
            batch_indices = targets[:, 0].long()
            _, counts = batch_indices.unique(return_counts=True)
            max_gt = counts.max().item()

        # Initialize output tensors
        gt_bboxes = torch.zeros(batch_size, max_gt, 4, device=device)
        gt_labels = torch.zeros(batch_size, max_gt, 1, dtype=torch.long, device=device)
        batch_idx = torch.zeros(batch_size, max_gt, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(batch_size, max_gt, 1, dtype=torch.bool, device=device)

        if targets.shape[0] == 0:
            return gt_bboxes, gt_labels, batch_idx, mask_gt

        # Fill in targets
        for b in range(batch_size):
            mask = targets[:, 0] == b
            if mask.sum() == 0:
                continue

            n = mask.sum()
            b_targets = targets[mask]

            # Convert xywh to xyxy and scale to pixels
            xy = b_targets[:, 2:4] * self.img_size
            wh = b_targets[:, 4:6] * self.img_size
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            bboxes = torch.cat([x1y1, x2y2], dim=-1)

            gt_bboxes[b, :n] = bboxes
            gt_labels[b, :n] = b_targets[:, 1:2].long()
            batch_idx[b, :n] = b_targets[:, 0:1].long()
            mask_gt[b, :n] = 1

        return gt_bboxes, gt_labels, batch_idx, mask_gt

    def _compute_empty_target_loss(self, predictions):
        """Handle empty targets"""
        device = predictions['cls'][0].device
        loss = 0.0
        for cls_pred in predictions['cls']:
            loss += cls_pred.abs().mean() * 0.001  # Small regularization
        return {
            'loss': torch.tensor(loss, device=device, requires_grad=True),
            'bbox_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }


def bbox2dist(anchor_points, bboxes, reg_max):
    """Transform bbox(xyxy) to dist(ltrb).

    Following ultralytics implementation: simple distance calculation with clamp.
    This assumes anchor_points and bboxes are in grid coordinates (not pixels).

    Args:
        anchor_points: (n_points, 2) anchor point coordinates (grid units)
        bboxes: (bs, n_points, 4) target bboxes in xyxy format (grid units)
        reg_max: maximum value for DFL (typically 16, so max distance is reg_max - 0.01)

    Returns:
        dist: (bs, n_points, 4) distances in ltrb format, clamped to [0, reg_max - 0.01]
    """
    bs = bboxes.shape[0]
    # anchor_points: (n_points, 2), need to expand to (bs, n_points, 2)
    anchor = anchor_points.unsqueeze(0).expand(bs, -1, -1)  # (bs, n_points, 2)

    x1y1, x2y2 = bboxes.chunk(2, dim=-1)  # Each: (bs, n_points, 2)
    lt = anchor - x1y1  # (bs, n_points, 2) left, top distances
    rb = x2y2 - anchor  # (bs, n_points, 2) right, bottom distances

    ltrb = torch.cat([lt, rb], dim=-1)  # (bs, n_points, 4)

    # Clamp to [0, reg_max - 0.01] following ultralytics
    # Note: reg_max - 0.01 (not reg_max - 1) ensures values stay within valid range
    return ltrb.clamp(0, reg_max - 0.01)
