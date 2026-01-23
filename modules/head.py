import torch
import torch.nn as nn


class Detect(nn.Module):
    """YOLOv3 Detect head for detection"""
    def __init__(self, nc=80, anchors=(), ch=()):
        """
        Args:
            nc: number of classes
            anchors: anchors for each detection layer
            ch: input channels for each detection layer (P3, P4, P5)
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (x, y, w, h, obj, classes)
        self.nl = len(anchors)  # number of detection layers (3)
        self.na = len(anchors[0]) // 2  # number of anchors per layer (3)
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.stride = torch.tensor([8., 16., 32.])  # strides for P3, P4, P5
        
        # Register anchors as buffer
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        
        # Output convolution layers for each detection scale
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        
    def forward(self, x):
        """
        Args:
            x: list of 3 feature maps [P3, P4, P5]
        Returns:
            if training: list of raw predictions [(bs, na, h, w, no), ...]
            if inference: (concatenated predictions, list of raw predictions)
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            # (bs, na*no, ny, nx) -> (bs, na, ny, nx, no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """Generate grid and anchor grid for predictions"""
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class DetectAnchorFree(nn.Module):
    """Anchor-free YOLO detection head (like YOLOv8/v11)

    Ultra-lightweight head with direct predictions (no intermediate layers).
    Similar to ultralytics implementation for minimal parameters.

    Outputs:
    - cls: (bs, nc, h, w) - classification logits
    - reg: (bs, 4, reg_max, h, w) - bbox distribution for DFL
    """

    def __init__(self, nc=80, reg_max=16, ch=()):
        """
        Args:
            nc: number of classes
            reg_max: number of distribution bins for DFL (default 16)
            ch: input channels for each detection layer (P3, P4, P5)
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers (3)
        self.reg_max = reg_max  # DFL distribution bins
        self.stride = torch.tensor([8., 16., 32.])  # strides for P3, P4, P5

        # Direct prediction heads (no intermediate layers, like ultralytics)
        self.cv3 = nn.ModuleList()  # cls output convs (1x1)
        self.cv4 = nn.ModuleList()  # reg output convs (1x1)

        for i in range(self.nl):
            # Classification: direct 1x1 conv from input to nc classes
            self.cv3.append(nn.Conv2d(ch[i], nc, 1))
            # Regression: direct 1x1 conv from input to 4*reg_max (DFL)
            self.cv4.append(nn.Conv2d(ch[i], 4 * self.reg_max, 1))

        # DFL projection (softmax over distribution bins)
        self.register_buffer(
            'proj',
            torch.arange(self.reg_max, dtype=torch.float32).view(1, 1, -1, 1, 1)
        )

        self.grid = [torch.zeros(1)] * self.nl

    def initialize_biases(self, img_size=640):
        """Initialize Detect head biases following ultralytics formula.

        This sets initial cls predictions to very low values (~0.00039 probability)
        to reduce initial loss and stabilize training.

        For regression (DFL), we initialize the bias to give predictions closer
        to the expected target range, reducing initial DFL loss.
        """
        import math

        for i, s in enumerate(self.stride):
            # Classification bias: initialize to low probability (~0.00039)
            # Following ultralytics formula
            bias_cls = math.log(5 / self.nc / (img_size / s) ** 2)
            self.cv3[i].bias.data[:] = bias_cls

            # Regression bias: initialize to push DFL predictions towards lower values
            # When bias=0, uniform distribution gives prediction of (reg_max-1)/2 = 15.5
            # Our targets are in range [0, ~30], so we need predictions centered around target mean
            # Setting bias more negative pushes distribution towards lower bins
            # For reg_max=32, bias ≈ -3.0 to -4.0 gives prediction ~7-12
            bias_reg = -3.5
            self.cv4[i].bias.data[:] = bias_reg

    def forward(self, x):
        """
        Args:
            x: list of 3 feature maps [P3, P4, P5]
        Returns:
            if training: {'cls': [(bs, nc, h, w), ...], 'reg': [(bs, 4*reg_max, h, w), ...]}
            if inference: (concatenated predictions, dict of outputs)
        """
        cls_outputs = []
        reg_outputs = []

        for i in range(self.nl):
            # Direct classification output (no intermediate layer)
            cls_output = self.cv3[i](x[i])  # (bs, nc, h, w)
            cls_outputs.append(cls_output)

            # Direct regression output (no intermediate layer)
            reg_output = self.cv4[i](x[i])  # (bs, 4*reg_max, h, w)
            reg_output = reg_output.reshape(-1, 4, self.reg_max, reg_output.shape[2], reg_output.shape[3])
            reg_outputs.append(reg_output)

        if not self.training:
            # Inference: convert DFL distribution to bbox coordinates
            return self._decode_inference(cls_outputs, reg_outputs, x)

        return {'cls': cls_outputs, 'reg': reg_outputs}

    def _decode_inference(self, cls_outputs, reg_outputs, x):
        """Decode predictions for inference"""
        z = []

        for i in range(self.nl):
            bs, _, ny, nx = cls_outputs[i].shape

            # Grid
            if self.grid[i].shape[2:4] != (ny, nx):
                d = cls_outputs[i].device
                yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
                self.grid[i] = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()

            # Classification: sigmoid for probability
            cls_score = cls_outputs[i].sigmoid()  # (bs, nc, h, w)

            # Regression: softmax over distribution, then project
            reg_dist = reg_outputs[i].softmax(2)  # (bs, 4, reg_max, h, w)
            reg_proj = (reg_dist * self.proj.to(reg_dist.device)).sum(2)  # (bs, 4, h, w)

            # Convert ltrb to xyxy
            # reg_proj: [left, top, right, bottom] distances in (bs, 4, h, w) format
            lt = reg_proj[:, :2]  # (bs, 2, h, w)
            rb = reg_proj[:, 2:]  # (bs, 2, h, w)

            # Grid: (1, 1, h, w, 2) -> (h, w, 2) for matching with lt/rb
            grid_hw = self.grid[i].squeeze(0).squeeze(0)  # (h, w, 2)

            # Expand grid to match batch size
            grid_expanded = grid_hw.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 2, h, w) -> (1, 2, h, w)

            # Compute xyxy coordinates
            x1y1 = grid_expanded - lt  # (bs, 2, h, w)
            x2y2 = grid_expanded + rb  # (bs, 2, h, w)

            # Convert to (bs, h, w, 4) format then to (bs, 4, h, w)
            xyxy = torch.cat([x1y1, x2y2], dim=1)  # (bs, 4, h, w)

            # Convert to cxcywh format
            xy = (x1y1 + x2y2) / 2 * self.stride[i]  # (bs, 2, h, w)
            wh = (x2y2 - x1y1) * self.stride[i]  # (bs, 2, h, w)

            # Combine: [bs, h, w, 4+nc]
            output = torch.cat([
                xy.permute(0, 2, 3, 1),   # (bs, h, w, 2)
                wh.permute(0, 2, 3, 1),   # (bs, h, w, 2)
                cls_score.permute(0, 2, 3, 1)  # (bs, h, w, nc)
            ], dim=-1)  # (bs, h, w, 4+nc)

            output = output.view(bs, -1, self.nc + 4)
            z.append(output)

        return torch.cat(z, 1), {'cls': cls_outputs, 'reg': reg_outputs}