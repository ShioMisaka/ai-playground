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
