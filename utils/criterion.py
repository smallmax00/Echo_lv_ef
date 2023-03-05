import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import region_edge_detector, GaussianRamp


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        intersection = predict * target
        score = (2. * intersection.sum(1) + self.smooth) / \
            (predict.sum(1) + target.sum(1) + self.smooth)
        loss = 1 - score.sum() / predict.size(0)
        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, weight) -> None:
        super(WeightedMSELoss, self).__init__()
        self.weight = weight.reshape(1, -1, 1)

    def forward(self, predict, target):
        return torch.sum(self.weight * (predict - target) ** 2)


class PolarIoU(nn.Module):
    def __init__(self):
        super(PolarIoU, self).__init__()

    def forward(self, pred, gt):
        cb = torch.stack((pred, gt))
        dmax, _ = cb.max(dim=0)
        dmin, _ = cb.min(dim=0)
        loss = torch.log(dmax.sum(dim=-1) / dmin.sum(dim=-1))

        return loss.mean()


class ContrastiveLoss(nn.Module):
    '''
    Contrastive loss based on Mask
    Shape:
        Input:
            (B, C, H, W) for Feature
            (B, 1, H, W) for Mask
    '''

    def __init__(self, temp=0.1, eps=1e-8, beta=0.9):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        self.beta = beta

    def forward(self, feat, anchor_Mask, pos_Mask, neg_Mask):
        ch = feat.shape[1]
        feat = feat.clone().permute(0, 2, 3, 1).reshape(-1, ch)
        anchor_Mask = anchor_Mask.reshape(-1)
        pos_Mask = pos_Mask.reshape(-1)
        neg_Mask = neg_Mask.reshape(-1)
        anchor_all = feat[anchor_Mask, :]
        pos_all = feat[pos_Mask, :]
        neg_all = feat[neg_Mask, :]

        # Sampling
        pos_feat = sampleFeat(pos_all)
        neg_feat = sampleFeat(neg_all)

        # if last_pos == None:
        #     pos_feat = pos_all.mean(0, keepdim=True)
        # else:
        #     pos_feat = last_pos * self.beta + pos_all.mean(0, keepdim=True) * (1 - self.beta)
        pos_feat = pos_all.mean(0, keepdim=True)

        pos = anchor_all @ pos_feat.t() / self.temp
        neg = anchor_all @ neg_feat.t() / self.temp
        loss = torch.exp(pos) / (torch.exp(pos) +
                                 torch.exp(neg).sum(-1, keepdim=True))
        return loss.mean(dim=-1)

        # if gt_Mask != None:
        #     gt_Mask = gt_Mask.reshape(-1)
        #     anchor_all = feat[gt_Mask.logical_and(~(gt.logical_and(anchor_Mask.logical_and(reg_Mask)))), :]
        #     pos_pair_all = feat[gt.logical_and(anchor_Mask.logical_and(reg_Mask)), :]
        #     neg_pair_all = feat[anchor_Mask.logical_and(reg_Mask.logical_and(~gt_Mask)), :]


def sampleFeat(feat, max_k=15000):
    k = min(feat.size(0), max_k)
    perm = torch.randperm(feat.size(0))
    idx = perm[:k]
    return feat[idx, :]


class Supervised_Loss(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.input_size = conf['input_size']
        self.seg_mid_size = (self.input_size[0] // 8, self.input_size[1] // 8)
        self.edge_kernel = conf['edge_kernel']
        self.edge_w = conf['edge_w']
        self.point_w = conf['point_w']
        self.point_mse_w = conf['point_mse_w']
        self.edge_con_w = conf['edge_con_w']
        self.point_con_w = conf['point_con_w']
        self.edge_contr_w = conf['edge_contr_w']
        self.point_contr_w = conf['point_contr_w']
        self.area_con_w = conf['area_con_w']
        self.semi_epoch = conf['semi_epoch']

        self.mask2edge = region_edge_detector(max_k=1, min_k=self.edge_kernel)
        self.mask2edge_outer = region_edge_detector(
            max_k=self.edge_kernel, min_k=1)

        self.dice_loss = BinaryDiceLoss()
        self.point_w_loss = nn.MSELoss()
        self.contr_loss = ContrastiveLoss()

        self.bce_loss = nn.BCELoss()

    def forward(self, seg_output, edge_output, point_feat, proj_feat, point_mask,
                region_l=None, edge_l=None, point_cb=None, epoch=None):
        device = seg_output.device
        loss_l = 0

        seg_edge = self.mask2edge(seg_output)

        region_loss = self.dice_loss(seg_output, region_l)
        edge_dc_loss = self.dice_loss(edge_output, edge_l)
        edge_ce_loss = self.bce_loss(edge_output, edge_l)

        point_l = point_cb

        point_loss = self.point_w_loss(point_feat, point_l)

        mask_ras_l_reg = self.dice_loss(point_mask, seg_output)
        edge_l_reg = self.dice_loss(seg_edge, edge_output)

        region_mid_loss = 0

        edge_con_w = GaussianRamp(self.edge_con_w, epoch, 4 * self.semi_epoch)
        point_con_w = GaussianRamp(
            self.point_con_w, epoch, 4 * self.semi_epoch)
        loss_l += region_loss + 0.1 * region_mid_loss \
            + self.edge_w * (edge_dc_loss + edge_ce_loss) \
            + self.point_w * point_loss \
            + edge_con_w * edge_l_reg + point_con_w * mask_ras_l_reg

        loss_list = [region_loss, region_mid_loss,
                     edge_dc_loss, edge_ce_loss,
                     point_loss,
                     edge_l_reg, mask_ras_l_reg]

        # Supervised Mask
        # Boundary
        with torch.no_grad():
            edge_output_Mask = F.interpolate(
                input=edge_output, size=self.seg_mid_size, mode='bilinear') > 0.5
            seg_edge_Mask = F.interpolate(
                input=seg_edge, size=self.seg_mid_size, mode='bilinear') > 0.5
            edge_l_Mask = F.interpolate(
                input=edge_l, size=self.seg_mid_size, mode='bilinear').bool()

            edge_pos_Mask = edge_l_Mask.logical_and(
                edge_output_Mask.logical_and(seg_edge_Mask))
            edge_anchor_Mask = edge_l_Mask.logical_and(~edge_pos_Mask)
            edge_neg_Mask = edge_output_Mask.logical_and(
                seg_edge_Mask.logical_and(~edge_pos_Mask))

        if torch.all(edge_pos_Mask == False) or torch.all(edge_anchor_Mask == False) or torch.all(edge_neg_Mask == False):
            edge_l_contr = torch.tensor(0.0, device=device)
        else:
            edge_l_contr = torch.utils.checkpoint.checkpoint(
                self.contr_loss, proj_feat, edge_anchor_Mask, edge_pos_Mask, edge_neg_Mask)
            edge_l_contr = edge_l_contr.mean()

        edge_contr_w = GaussianRamp(
            self.edge_contr_w, epoch, 60 + 20, start_epoch=60)
        loss_l += edge_contr_w * edge_l_contr

        loss_list.append(edge_l_contr)

        return loss_l, loss_list


class Unsupervised_Loss(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.input_size = conf['input_size']
        self.seg_mid_size = (self.input_size[0] // 8, self.input_size[1] // 8)
        self.edge_kernel = conf['edge_kernel']
        self.edge_w = conf['edge_w']
        self.point_w = conf['point_w']
        self.edge_con_w = conf['edge_con_w']
        self.point_con_w = conf['point_con_w']
        self.edge_contr_w = conf['edge_contr_w']
        self.point_contr_w = conf['point_contr_w']
        self.area_con_w = conf['area_con_w']
        self.semi_epoch = conf['semi_epoch']

        self.mask2edge = region_edge_detector(max_k=1, min_k=self.edge_kernel)
        self.mask2edge_outer = region_edge_detector(
            max_k=self.edge_kernel, min_k=1)

        self.dice_loss = BinaryDiceLoss()
        self.contr_loss = ContrastiveLoss()

        self.bce_loss = nn.BCELoss()

    def forward(self, seg_output_ul, edge_output_ul, proj_feat_ul, point_mask_ul, epoch):
        loss_ul = 0
        loss_list = []
        device = seg_output_ul.device

        seg_edge_ul = self.mask2edge(seg_output_ul)

        mask_ras_ul_reg = self.dice_loss(point_mask_ul, seg_output_ul)
        edge_ul_reg = self.dice_loss(seg_edge_ul, edge_output_ul)

        edge_con_w = GaussianRamp(self.edge_con_w, epoch, 4 * self.semi_epoch)
        point_con_w = GaussianRamp(
            self.point_con_w, epoch, 4 * self.semi_epoch)
        loss_ul += edge_con_w * edge_ul_reg + point_con_w * mask_ras_ul_reg
        loss_list.append(edge_ul_reg)
        loss_list.append(mask_ras_ul_reg)

        # Unsupervised Mask
        # Boundary
        with torch.no_grad():
            seg_output_Mask_ul = F.interpolate(
                input=seg_output_ul, size=self.seg_mid_size, mode='bilinear') > 0.5
            edge_output_Mask_ul = F.interpolate(
                input=edge_output_ul, size=self.seg_mid_size, mode='bilinear') > 0.5
            seg_edge_Mask_ul = F.interpolate(
                input=seg_edge_ul, size=self.seg_mid_size, mode='bilinear') > 0.5

            seg_edge_Mask_outer = self.mask2edge_outer(
                seg_edge_Mask_ul.float()).bool()
            seg_output_Mask_outer = self.mask2edge_outer(
                seg_output_Mask_ul.float()).bool()

            # center region of seg output
            edge_pos_Mask_ul = seg_output_Mask_ul.logical_and(
                ~seg_edge_Mask_ul)
            edge_anchor_Mask_ul = edge_output_Mask_ul.logical_and(
                seg_edge_Mask_ul)
            edge_neg_Mask_ul = seg_edge_Mask_outer.logical_and(
                seg_output_Mask_outer)

        if torch.all(edge_pos_Mask_ul == False) or torch.all(edge_anchor_Mask_ul == False) or torch.all(edge_neg_Mask_ul == False):
            edge_ul_contr = torch.tensor(0.0, device=device)
        else:
            edge_ul_contr = torch.utils.checkpoint.checkpoint(
                self.contr_loss, proj_feat_ul, edge_anchor_Mask_ul, edge_pos_Mask_ul, edge_neg_Mask_ul)
            edge_ul_contr = edge_ul_contr.mean()

        edge_contr_w = GaussianRamp(
            self.edge_contr_w, epoch, 200 + 20, start_epoch=200)
        loss_ul += edge_contr_w * edge_ul_contr
        loss_list.append(edge_ul_contr)

        return loss_ul, loss_list
