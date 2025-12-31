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