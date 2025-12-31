import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    """YOLOv3 Loss Function"""
    
    def __init__(self, num_classes, anchors, img_size=640):
        """
        Args:
            num_classes: 类别数量
            anchors: 每个尺度的anchor boxes [[w1,h1,w2,h2,...], [...], [...]]
            img_size: 输入图像尺寸
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
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
    
    def forward(self, predictions, targets):
        """
        计算loss
        
        Args:
            predictions: 模型输出列表
            targets: ground truth [num_boxes, 6] - (batch_idx, class_id, x, y, w, h) 归一化坐标
        
        Returns:
            total_loss: 总损失
        """
        device = predictions[0].device
        
        # 初始化损失
        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        # 如果没有目标，只计算noobj loss
        if targets.shape[0] == 0:
            for scale_idx, pred in enumerate(predictions):
                batch_size = pred.shape[0]
                num_anchors = len(self.anchors[scale_idx])
                grid_size = pred.shape[2]
                
                pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size)
                pred = pred.permute(0, 1, 3, 4, 2).contiguous()
                pred_conf = pred[..., 4:5]
                
                target_obj = torch.zeros_like(pred_conf)
                loss_obj += self.lambda_noobj * self.bce_loss(pred_conf, target_obj)
            
            # 归一化
            total_pixels = sum([p.shape[0] * len(self.anchors[i]) * p.shape[2] * p.shape[3] 
                               for i, p in enumerate(predictions)])
            return loss_obj / max(total_pixels, 1)
        
        num_targets = 0
        num_total_pixels = 0
        
        # 对每个尺度计算loss
        for scale_idx, pred in enumerate(predictions):
            batch_size = pred.shape[0]
            num_anchors = len(self.anchors[scale_idx])
            grid_size = pred.shape[2]
            
            num_total_pixels += batch_size * num_anchors * grid_size * grid_size
            
            # 重塑预测
            pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # 获取预测的各个部分，应用激活函数
            pred_xy = torch.sigmoid(pred[..., 0:2])      # sigmoid激活中心坐标
            pred_wh = pred[..., 2:4]                      # 宽高不激活（后续用exp）
            pred_conf = pred[..., 4:5]                    # 置信度（用logits）
            pred_cls = pred[..., 5:]                      # 类别（用logits）
            
            # 构建目标张量
            target_obj = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
            target_box = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 4, device=device)
            target_cls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, self.num_classes, device=device)
            
            # 为每个目标分配
            for target in targets:
                if target.sum() == 0:
                    continue
                    
                batch_idx = int(target[0])
                if batch_idx >= batch_size:
                    continue
                    
                class_id = int(target[1])
                if class_id >= self.num_classes:
                    continue
                
                x, y, w, h = target[2:6]
                
                # 检查坐标有效性
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    continue
                
                # 转换到当前grid尺度
                gx = x * grid_size
                gy = y * grid_size
                gw = w * grid_size
                gh = h * grid_size
                
                # 找到对应的grid cell
                gi = int(gx)
                gj = int(gy)
                
                # 确保索引在有效范围内
                gi = min(gi, grid_size - 1)
                gj = min(gj, grid_size - 1)
                
                # 计算与每个anchor的IoU
                gt_box = torch.tensor([0, 0, gw, gh], device=device)
                anchor_boxes = torch.cat([
                    torch.zeros(num_anchors, 2, device=device),
                    self.anchors[scale_idx].to(device) / (self.img_size / grid_size)
                ], dim=1)
                
                # 计算IoU
                ious = self.bbox_iou(gt_box.unsqueeze(0), anchor_boxes)
                best_anchor = torch.argmax(ious)
                
                # 设置目标
                target_obj[batch_idx, best_anchor, gj, gi, 0] = 1.0
                target_box[batch_idx, best_anchor, gj, gi, 0] = gx - gi  # x offset [0,1]
                target_box[batch_idx, best_anchor, gj, gi, 1] = gy - gj  # y offset [0,1]
                
                # 使用log space来避免数值问题
                anchor_w = self.anchors[scale_idx][best_anchor, 0] / (self.img_size / grid_size)
                anchor_h = self.anchors[scale_idx][best_anchor, 1] / (self.img_size / grid_size)
                target_box[batch_idx, best_anchor, gj, gi, 2] = torch.log(gw / (anchor_w + 1e-16) + 1e-16)
                target_box[batch_idx, best_anchor, gj, gi, 3] = torch.log(gh / (anchor_h + 1e-16) + 1e-16)
                
                target_cls[batch_idx, best_anchor, gj, gi, class_id] = 1.0
                num_targets += 1
            
            # 计算mask
            obj_mask = target_obj.squeeze(-1) > 0
            noobj_mask = target_obj.squeeze(-1) == 0
            
            # Box loss (只计算有目标的位置)
            if obj_mask.sum() > 0:
                # xy loss
                loss_box += self.mse_loss(
                    pred_xy[obj_mask],
                    target_box[..., :2][obj_mask]
                )
                
                # wh loss (使用log space)
                loss_box += self.mse_loss(
                    pred_wh[obj_mask],
                    target_box[..., 2:][obj_mask]
                )
            
            # Objectness loss
            if obj_mask.sum() > 0:
                loss_obj += self.bce_loss(
                    pred_conf[obj_mask],
                    target_obj[obj_mask]
                )
            
            if noobj_mask.sum() > 0:
                loss_obj += self.lambda_noobj * self.bce_loss(
                    pred_conf[noobj_mask],
                    target_obj[noobj_mask]
                )
            
            # Class loss
            if obj_mask.sum() > 0:
                loss_cls += self.bce_loss(
                    pred_cls[obj_mask],
                    target_cls[obj_mask]
                )
        
        # 归一化loss
        num_targets = max(num_targets, 1)
        loss_box = loss_box / num_targets
        loss_obj = loss_obj / num_total_pixels
        loss_cls = loss_cls / num_targets
        
        # 总loss
        total_loss = (
            self.lambda_coord * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )
        
        # 检查NaN
        if torch.isnan(total_loss):
            print(f"Warning: NaN detected! box_loss={loss_box}, obj_loss={loss_obj}, cls_loss={loss_cls}")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss
    
    def bbox_iou(self, box1, box2, eps=1e-7):
        """计算IoU"""
        # 转换为 (x1, y1, x2, y2) 格式
        b1_x1 = box1[:, 0:1] - box1[:, 2:3] / 2
        b1_y1 = box1[:, 1:2] - box1[:, 3:4] / 2
        b1_x2 = box1[:, 0:1] + box1[:, 2:3] / 2
        b1_y2 = box1[:, 1:2] + box1[:, 3:4] / 2
        
        b2_x1 = box2[:, 0:1].T - box2[:, 2:3].T / 2
        b2_y1 = box2[:, 1:2].T - box2[:, 3:4].T / 2
        b2_x2 = box2[:, 0:1].T + box2[:, 2:3].T / 2
        b2_y2 = box2[:, 1:2].T + box2[:, 3:4].T / 2
        
        # 交集
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 并集
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area.T - inter_area + eps
        
        return inter_area / union_area
    
    def bbox_iou(self, box1, box2, eps=1e-7):
        """
        计算两个box的IoU
        box1: [N, 4] (x, y, w, h) 中心坐标格式
        box2: [M, 4] (x, y, w, h) 中心坐标格式
        返回: [N, M] IoU矩阵
        """
        # 转换为 (x1, y1, x2, y2) 格式
        b1_x1 = box1[:, 0:1] - box1[:, 2:3] / 2
        b1_y1 = box1[:, 1:2] - box1[:, 3:4] / 2
        b1_x2 = box1[:, 0:1] + box1[:, 2:3] / 2
        b1_y2 = box1[:, 1:2] + box1[:, 3:4] / 2
        
        b2_x1 = box2[:, 0:1].T - box2[:, 2:3].T / 2
        b2_y1 = box2[:, 1:2].T - box2[:, 3:4].T / 2
        b2_x2 = box2[:, 0:1].T + box2[:, 2:3].T / 2
        b2_y2 = box2[:, 1:2].T + box2[:, 3:4].T / 2
        
        # 交集
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 并集
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area.T - inter_area + eps
        
        return inter_area / union_area