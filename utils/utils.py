import sys  # NOQA: E402
sys.path.append("../BoundaryFormer/projects/BoundaryFormer")  # NOQA: E402
import pdb
from torchvision.transforms.functional import gaussian_blur
from boundary_former.layers.diff_ras.polygon import SoftPolygon
import numpy as np
import torch.nn as nn
import torch


def saveModel(model, optimizer, scheduler, epoch_num, save_mode_path):
    torch.save({
        'epoch': epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_mode_path)


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def GaussianRamp(max_weight, epoch, max_epoch, start_epoch=0.0):
    if epoch == None:
        return max_weight
    if epoch < start_epoch:
        return 0 * max_weight

    epoch = np.clip(epoch, start_epoch, max_epoch)
    x = 1.0 - (epoch - start_epoch) / (max_epoch - start_epoch)
    ramps = float(np.exp(-5.0 * x * x))

    weight = max_weight * ramps
    return weight


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


class Polar2Mask(nn.Module):
    def __init__(self, num_kp=12):
        super(Polar2Mask, self).__init__()
        self.num_kp = num_kp
        self.rasterizer = SoftPolygon(mode="mask")

        degrees = np.linspace(-np.pi, np.pi, self.num_kp + 1)
        dtheta_np = np.vstack((np.cos(degrees)[:-1], np.sin(degrees)[:-1]))
        self.dtheta = torch.tensor(
            dtheta_np, dtype=torch.float, device='cuda').permute(1, 0)

    def forward(self, center, dist, output_size):
        dxdy = dist.unsqueeze(-1) * self.dtheta
        con_vertices = dxdy + center.unsqueeze(1)
        if torch.any(con_vertices <= 0):
            B = con_vertices.shape[0]
            lb = output_size[0] - 10
            rt = output_size[0] + 10
            con_vertices = torch.tensor(
                [[lb, lb], [lb, rt], [rt, rt], [rt, lb]], dtype=torch.float, device='cuda')
            con_vertices = con_vertices.repeat(B, 1, 1)
        mask = self.rasterizer(con_vertices.contiguous(),
                               output_size[0], output_size[1], 1.0)

        return mask, con_vertices


class region_edge_detector(nn.Module):
    """
    Extract edge from given mask.
    """

    def __init__(self, max_k=1, min_k=9):  # width = 4
        super(region_edge_detector, self).__init__()
        self.max_pooling_1 = nn.MaxPool2d(min_k, 1, min_k//2)
        self.max_pooling_2 = nn.MaxPool2d(max_k, 1, max_k//2)

    def forward(self, x):
        eros = -self.max_pooling_1(-x)
        expa = self.max_pooling_2(x)
        edge = expa - eros
        return edge


def dist2con(center, dist, num_kp):
    device = center.device

    degrees = np.linspace(-np.pi, np.pi, num_kp + 1)
    dtheta_np = np.vstack((np.cos(degrees)[:-1], np.sin(degrees)[:-1]))
    dtheta = torch.tensor(dtheta_np, dtype=torch.float,
                          device=device).permute(1, 0)

    dxdy = dist.unsqueeze(-1) * dtheta
    output_con = dxdy + center.unsqueeze(1)

    return output_con


def EstAxis(point_ori: torch.Tensor):
    """
    Estimate length of ventricle from boundary points
    point_ori: B * Channel * XY
    No terminal points
    """
    deltaXY = point_ori[:, 2:20] - torch.flip(point_ori, dims=(1,))[:, 2:20]
    deltaXY = deltaXY.sum(dim=1)
    deltaXY = deltaXY / ((deltaXY ** 2).sum(dim=-1, keepdim=True) ** (1/2))
    axis_vector = deltaXY @ torch.tensor([[0, 1],
                                          [-1, 0]], device='cuda').float()
    top = (point_ori[:, 1] + point_ori[:, -2]) / 2
    bottom = (point_ori[:, 20] + point_ori[:, 21]) / 2
    axis = torch.abs(((top - bottom) * axis_vector).sum(dim=-1) * (20/19))

    return axis


def EstEV(point_ori: torch.Tensor):
    """
    Estimate EV from boundary points
    point_ori: B * Channel * XY
    No terminal points
    """
    line = ((point_ori - torch.flip(point_ori, dims=(1,)))
            ** 2).sum(dim=-1) ** (1/2)
    axis = EstAxis(point_ori)
    ev = (line[:, 1:20]**2 + line[:, 1:20]*line[:, 2:21] +
          line[:, 2:21]**2).sum(dim=-1) * axis

    return ev


def pointAttention(coords, ras_mask):
    '''
    coords: Tensor; B * num_kp * 2
    rasMask: Tensor; B * W * H
    '''
    B = ras_mask.shape[0]
    H = ras_mask.shape[-1]
    device = ras_mask.device
    with torch.no_grad():
        coords = coords.long()
        if (torch.any(coords < 0) or torch.any(coords > H - 1)):
            coords = torch.tensor([[H // 4, H // 4], [H // 4, 3 * H // 4], [
                                  3 * H // 4, 3 * H // 4], [3 * H // 4, H // 4]], device=device).repeat(B, 1, 1)
        flatten_coords = coords[:, :, 1] * H + coords[:, :, 0]
        coords_mask = torch.zeros_like(ras_mask, device=device).reshape(B, -1)
        coords_mask.scatter_(-1, flatten_coords, 1)
    atten_mask = ras_mask.reshape(B, -1) * coords_mask

    return atten_mask


def Point2Mask(rasterizer, coords, output_size):
    '''
    rasterizer: Function
    coords: Tensor; B * C * num_kp * 2
    output_size: Tuple; (H, H)
    '''

    B = coords.shape[0]
    C = coords.shape[1]
    H = output_size[0]
    device = coords.device
    if (torch.any(coords < 0) or torch.any(coords > H - 1)):
        coords = torch.tensor([[H // 4, H // 4], [H // 4, 3 * H // 4], [3 * H // 4, 3 * H // 4], [
                              3 * H // 4, H // 4]], dtype=torch.float, device=device).repeat(B, C, 1, 1)
    ras_mask = torch.zeros((B, C, H, H), device=device)
    try:
        for i in range(C):
            ras_mask[:, i, :, :] = rasterizer(coords[:, i, :, :], H, H, 1.0)
    except:
        pdb.set_trace()
    return ras_mask
