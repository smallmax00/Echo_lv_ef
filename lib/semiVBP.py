import sys  # NOQA: E402
sys.path.append("../BoundaryFormer/projects/BoundaryFormer")  # NOQA: E402
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s
from boundary_former.layers.diff_ras.polygon import SoftPolygon
from utils.utils import Point2Mask, EstAxis
from utils.criterion import BinaryDiceLoss, Supervised_Loss, Unsupervised_Loss

"""
Model of semiVBP

Please install kornia from https://github.com/kornia/kornia
and BoundaryFormer from https://github.com/mlpc-ucsd/BoundaryFormer

Res2Net_v1b is imported from original lib/
"""


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Projector(nn.Module):
    def __init__(self, num_in, num_out):
        super(Projector, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.conv1 = nn.Conv2d(self.num_in, self.num_in,
                               kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.num_in, self.num_out,
                               kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class EDGModule(nn.Module):
    def __init__(self, channel, output_channel=1):
        super(EDGModule, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(
            2 * channel, 2 * channel, 3, padding=1)

        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)

        self.conv5 = nn.Conv2d(4 * channel, output_channel, 1)

    def forward(self, x1, x2, x3):  # 16x16, 32x32, 64x64
        up_x1 = self.conv_upsample1(self.upsample(x1))
        conv_x2 = self.conv_1(x2)
        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))

        up_x2 = self.conv_upsample2(self.upsample(x2))
        conv_x3 = self.conv_2(x3)
        cat_x3 = self.conv_concat3(torch.cat((up_x2, conv_x3), 1))

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2))
        conv_cat_x3 = self.conv_3(cat_x3)
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, conv_cat_x3), 1))
        x = self.conv5(cat_x4)
        return x, cat_x4


class PDGModule(EDGModule):
    def __init__(self, input_size, channel, output_channel=1, final_channel=24):
        super(PDGModule, self).__init__(channel, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        pooled_area = (input_size[0] // 2) * (input_size[1] // 2)

        self.fc_1 = nn.Linear(pooled_area * output_channel, 1024)
        self.fc_2 = nn.Linear(1024, 128)
        self.fc_3 = nn.Linear(128, final_channel)

    def forward(self, x1, x2, x3):  # 16x16, 32x32, 64x64
        x, _ = super(PDGModule, self).forward(x1, x2, x3)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x


class Extractor(nn.Module):
    """
    Extractor of semiVBP
    Return the logit of region, edge, standardized point coordinates, contrastive feature, point rasterized mask, and axis of point
    Args:
        point_avg: averged coordinated of points.
        point_std: std of coordinated of points.
        input_size: Size of input in form of (H, W).
        mid_channel: Number of channels in middle layers.
        output_channel: Number of channels produced by the network.
        num_kp: Number of keypoints.

    Shape:
        Input: (B, Cin, H, W)
        Output: (B, Cout, H, W) for all outputs # Cout > 1 not ready
    """

    def __init__(self, point_avg, point_std, input_size, output_channel=1, mid_channel=64, proj_channel=64, num_kp=12):
        super(Extractor, self).__init__()
        self.input_size = input_size
        self.edge_size = (input_size[0] // 4, input_size[1] // 4)
        self.seg_size = (input_size[0] // 8, input_size[1] // 8)
        self.num_kp = num_kp
        self.output_channel = output_channel

        self.point_avg = torch.tensor(
            point_avg, dtype=torch.float, device='cuda').reshape(self.output_channel, 1, 2)
        self.point_std = torch.tensor(
            point_std, dtype=torch.float, device='cuda').reshape(self.output_channel, 1, 2)

        self.resnet = res2net50_v1b_26w_4s(pretrained=False)

        self.rfb2_1 = BasicConv2d(256, mid_channel, 1)
        self.rfb3_1 = BasicConv2d(512, mid_channel, 1)
        self.rfb4_1 = BasicConv2d(1024, mid_channel, 1)
        self.rfb5_1 = BasicConv2d(2048, mid_channel, 1)

        self.edge_layer = EDGModule(mid_channel, output_channel)

        self.seg_layer = EDGModule(mid_channel, output_channel)
        self.projector = Projector(mid_channel * 4, proj_channel)

        self.point_layer = PDGModule(
            self.edge_size, mid_channel, output_channel, output_channel * self.num_kp * 2)

        self.rasterizer = SoftPolygon(mode="mask")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x1_rfb = self.rfb2_1(x1)
        x2_rfb = self.rfb3_1(x2)
        x3_rfb = self.rfb4_1(x3)
        x4_rfb = self.rfb5_1(x4)

        B = x.shape[0]
        edge_feat, _ = self.edge_layer(x3_rfb, x2_rfb, x1_rfb)
        point_feat = self.point_layer(x3_rfb, x2_rfb, x1_rfb)

        point_coord = point_feat.reshape(B, self.output_channel, -1, 2)
        point_coord_ori = point_coord * self.point_std + self.point_avg

        # Original size
        ras_mask = Point2Mask(
            self.rasterizer, point_coord_ori[:, :, 1:-1, :], self.input_size)

        seg_feat, seg_feat_h = self.seg_layer(x4_rfb, x3_rfb, x2_rfb)
        # Upscale to original size
        seg_feat = F.interpolate(
            input=seg_feat, size=self.input_size, mode='bilinear').sigmoid_()
        edge_feat = F.interpolate(
            input=edge_feat, size=self.input_size, mode='bilinear').sigmoid_()
        point_mask = F.interpolate(
            input=ras_mask, size=self.input_size, mode='bilinear')

        # calc ev
        axis = EstAxis(point_coord_ori[:, 0])
        ev = seg_feat[:, 0].sum(dim=(-1, -2)) ** 2 / axis

        proj_feat = self.projector(seg_feat_h)
        proj_feat = F.normalize(proj_feat, 2, 1)

        return seg_feat, edge_feat, point_coord, proj_feat, point_mask, ev


class semiVBP(nn.Module):
    def __init__(self, conf):
        super(semiVBP, self).__init__()
        self.input_size = conf['input_size']
        self.seg_mid_size = (self.input_size[0] // 8, self.input_size[1] // 8)
        self.point_avg = conf['point_avg']
        self.point_std = conf['point_std']
        self.output_channel = conf['output_channel']
        self.mid_channel = conf['mid_channel']
        self.proj_channel = conf['proj_channel']
        self.num_kp = conf['num_kp']
        self.edge_kernel = conf['edge_kernel']
        self.ef_w = conf['ef_w']

        self.ul_w = conf['ul_w']

        self.smallExtractor = Extractor(
            self.point_avg[0], self.point_std[0], self.input_size, self.output_channel, self.mid_channel, self.proj_channel, self.num_kp)
        self.largeExtractor = Extractor(
            self.point_avg[1], self.point_std[1], self.input_size, self.output_channel, self.mid_channel, self.proj_channel, self.num_kp)
        self.supervisedLoss = Supervised_Loss(conf)
        self.unsupervisedLoss = Unsupervised_Loss(conf)
        self.dice_loss = BinaryDiceLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_l=None,
                region_l=None, edge_l=None, point_cb=None, ef_l=None,
                x_ul=None, epoch=None, iter_num=None):
        if not self.training:
            smallSeg_output, smallEdge_output, small_point_feat, small_proj, _, pred_esv = self.smallExtractor(
                x_l[:, :3])
            largeSeg_output, largeEdge_output, large_point_feat, large_proj, _, pred_edv = self.largeExtractor(
                x_l[:, 3:])

            if region_l != None:
                smallRegion_loss = self.dice_loss(
                    smallSeg_output, region_l[:, [0]])
                largeRegion_loss = self.dice_loss(
                    largeSeg_output, region_l[:, [1]])
            else:
                smallRegion_loss = torch.tensor(0.0, device='cuda')
                largeRegion_loss = torch.tensor(0.0, device='cuda')
            if ef_l != None:
                pred_ef = (1 - (pred_esv / pred_edv)) * 100
                loss_ef = self.mse_loss(pred_ef.squeeze(), ef_l.squeeze())
            else:
                pred_ef = torch.tensor(0.0, device='cuda')
                loss_ef = torch.tensor(0.0, device='cuda')

            return smallRegion_loss, largeRegion_loss, loss_ef, smallSeg_output, largeSeg_output, small_point_feat, large_point_feat, pred_esv, pred_edv, pred_ef, smallEdge_output, largeEdge_output, small_proj, large_proj

        loss_tot = 0
        smallLoss_l = smallLoss_ul = largeLoss_l = largeLoss_ul = 0
        smallLoss_list_l = smallLoss_list_ul = largeLoss_list_l = largeLoss_list_ul = []

        # small
        seg_feat, edge_feat, point_feat, proj_feat, point_mask, pred_esv = self.smallExtractor(
            x_l[:, :3])
        smallLoss_l, smallLoss_list_l = self.supervisedLoss(seg_feat, edge_feat, point_feat, proj_feat, point_mask,
                                                            region_l[:, [0]], edge_l[:, [0]], point_cb[:, [0]], epoch)
        if x_ul != None:
            seg_feat_ul, edge_feat_ul, point_feat_ul, proj_feat_ul, point_mask_ul, pred_esv_ul = self.smallExtractor(
                x_ul[:, :3])
            smallLoss_ul, smallLoss_list_ul = self.unsupervisedLoss(seg_feat_ul, edge_feat_ul, proj_feat_ul, point_mask_ul,
                                                                    epoch)

        smallLoss = smallLoss_l + self.ul_w * smallLoss_ul
        smallLoss_list = smallLoss_list_l + smallLoss_list_ul

        # large
        seg_feat, edge_feat, point_feat, proj_feat, point_mask, pred_edv = self.largeExtractor(
            x_l[:, 3:])
        largeLoss_l, largeLoss_list_l = self.supervisedLoss(seg_feat, edge_feat, point_feat, proj_feat, point_mask,
                                                            region_l[:, [1]], edge_l[:, [1]], point_cb[:, [1]], epoch)
        if x_ul != None:
            seg_feat_ul, edge_feat_ul, point_feat_ul, proj_feat_ul, point_mask_ul, pred_edv_ul = self.largeExtractor(
                x_ul[:, 3:])
            largeLoss_ul, largeLoss_list_ul = self.unsupervisedLoss(seg_feat_ul, edge_feat_ul, proj_feat_ul, point_mask_ul,
                                                                    epoch)

        largeLoss = largeLoss_l + self.ul_w * largeLoss_ul
        largeLoss_list = largeLoss_list_l + largeLoss_list_ul

        # ef
        loss_ef = 0.0
        if (x_ul != None) and (ef_l != None):
            pred_ef = (1 - (pred_esv / pred_edv)) * 100

            loss_ef = self.mse_loss(pred_ef.squeeze(), ef_l)
            loss_tot += self.ef_w * loss_ef

        loss_tot = smallLoss + largeLoss
        loss_list = [smallLoss_list[i] + largeLoss_list[i]
                     for i in range(len(smallLoss_list))]

        return loss_tot, loss_list, smallLoss_list[0], largeLoss_list[0], loss_ef
