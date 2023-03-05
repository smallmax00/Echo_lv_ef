import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from scipy.ndimage import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from utils.plot_boundary import plot_point, plot_ras_mask, plot_tsne
from utils.dataset import BaseDataSets
from scipy.stats import pearsonr
from lib.semiVBP import semiVBP
import cv2
import multiprocessing as mp


def calculate_metric_percase(pred, gt):
    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    dice = dice_coef(gt, pred)
    jc = jaccard_score(gt.flatten(), pred.flatten())
    bc = balanced_accuracy_score(gt.flatten(), pred.flatten())
    hd = hd95(gt, pred)

    return dice, jc, bc, hd


def hd95(result, reference):
    '''
    95th percentile of the Hausdorff Distance.
    https://loli.github.io/medpy/_modules/medpy/metric/binary.html
    '''
    def surface_distances(result, reference, connectivity=1):
        """
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        """
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))

        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)

        # test for emptiness
        if 0 == np.count_nonzero(result):
            raise RuntimeError(
                'The first supplied array does not contain any binary object.')
        if 0 == np.count_nonzero(reference):
            raise RuntimeError(
                'The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(
            result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(
            reference, structure=footprint, iterations=1)

        # compute average surface distance
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border)
        sds = dt[result_border]

        return sds
    try:
        hd1 = surface_distances(result, reference)
        hd2 = surface_distances(reference, result)
        hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    except:
        hd95 = 8
    return hd95


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../semiVBP_Datasets/Echonet', help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='h5py_point_norm_scaled',
                    help='name of dataset')
parser.add_argument('--model_dir', type=str,
                    default='EstAL-contrAD', help='directory of model')
parser.add_argument('--testlist', type=str,
                    help='path of test list')
parser.add_argument('--model_name', type=str,
                    help='saved model filename')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--semi_epoch', type=int, default=20,
                    help='start epoch of semi-supervise')
parser.add_argument("--cuda_id", type=str,
                    default='0', help='cuda ID')
parser.add_argument('--viz', type=bool,
                    default=False, help='save pred gt etc')
parser.add_argument('--ras', type=bool,
                    default=False, help='save rasterization of pred')
parser.add_argument('--tsne', type=bool,
                    default=False, help='save tsne of pred')
parser.add_argument('--CI', type=bool,
                    default=False, help='confidence interval of metrics')
args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
# device_ids = [0, 1, 2]

conf = {
    'input_size': tuple(args.patch_size),
    # Echonet small, large
    'point_avg': [[134.63064776, 110.8778278],
                  [137.666924, 115.07382282]],
    'point_std': [[21.5190072, 33.95478734],
                  [28.30969132, 38.43371816]],
    'output_channel': 1,
    'mid_channel': 64,
    'proj_channel': 64,
    'num_kp': 42,
    'edge_kernel': 9,
    'edge_w': 0.1,
    'point_w': 0.1,
    'point_mse_w': [],
    'edge_con_w': 0.05,  # 0.05
    'point_con_w': 0.05,  # 0.05
    'area_con_w': 1e-9,
    'edge_contr_w': 0.01,  # 0.01
    'point_contr_w': 0.1,  # not used
    'ef_w': 0.01,  # 0.01
    'ul_w': 0.5,  # 0.5
    'semi_epoch': args.semi_epoch
}

point_avg = np.array(conf['point_avg'])
point_std = np.array(conf['point_std'])


def infer(args, snapshot_path):

    db_val = BaseDataSets(base_dir=args.root_path, split="test",
                          dataset_name=args.dataset, testlist=args.testlist)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    torch.cuda.empty_cache()

    model = semiVBP(conf)
    model = model.cuda()

    checkpoint = torch.load(snapshot_path + '/' + args.model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        total_metric_large = 0.0
        total_metric_small = 0.0

        large_list = []
        small_list = []
        ef_list = []
        large_point_list = []
        small_point_list = []
        name_list = []
        vef_pred_list, vef_gt_list = [], []

        if args.tsne:
            pool = mp.Pool(processes=20)

        for i_batch, (sampled_batch, name) in enumerate(tqdm(valloader)):
            name = name[0]
            volume_batch = sampled_batch['image'].float().cuda()
            region_gt = sampled_batch['region'].float()
            edge_gt = sampled_batch['edge'].float()
            point_gt = sampled_batch['point'].float()
            ef_gt = sampled_batch['ef'].float().cuda()

            _, _, ef_mse, small_seg, large_seg, small_point, large_point, _, _, ef_pred, small_edge, large_edge, small_proj, large_proj = model(
                x_l=volume_batch, ef_l=ef_gt)

            small_seg = small_seg.cpu().detach().numpy().squeeze()
            large_seg = large_seg.cpu().detach().numpy().squeeze()
            small_edge = small_edge.cpu().detach().numpy().squeeze()
            large_edge = large_edge.cpu().detach().numpy().squeeze()
            small_proj = small_proj.cpu().detach().numpy().squeeze()
            large_proj = large_proj.cpu().detach().numpy().squeeze()

            small_map = (small_seg > 0.5).astype(np.uint8)
            large_map = (large_seg > 0.5).astype(np.uint8)
            small_edge = (small_edge > 0.5).astype(np.uint8)
            large_edge = (large_edge > 0.5).astype(np.uint8)

            small_point = small_point.cpu().detach().numpy().squeeze()
            large_point = large_point.cpu().detach().numpy().squeeze()
            small_point = small_point * point_std[0] + point_avg[0]
            large_point = large_point * point_std[1] + point_avg[1]

            small_map_gt = region_gt[0, 0].numpy(
            ).squeeze().astype(np.uint8)
            large_map_gt = region_gt[0, 1].numpy(
            ).squeeze().astype(np.uint8)
            small_edge_gt = edge_gt[0, 0].numpy(
            ).squeeze().astype(np.uint8)
            large_edge_gt = edge_gt[0, 1].numpy(
            ).squeeze().astype(np.uint8)
            small_point_gt = point_gt[0, 0].numpy().squeeze()
            large_point_gt = point_gt[0, 1].numpy().squeeze()
            small_point_gt = small_point_gt * point_std[0] + point_avg[0]
            large_point_gt = large_point_gt * point_std[1] + point_avg[1]

            single_metric_small = calculate_metric_percase(
                small_map, small_map_gt)
            total_metric_small += np.asarray(single_metric_small)
            single_metric_large = calculate_metric_percase(
                large_map, large_map_gt)
            total_metric_large += np.asarray(single_metric_large)

            large_list.append(single_metric_large)
            small_list.append(single_metric_small)
            large_point_list.append(
                [mean_squared_error(large_point, large_point_gt, squared=False), mean_absolute_error(large_point, large_point_gt)])
            small_point_list.append(
                [mean_squared_error(small_point, small_point_gt, squared=False), mean_absolute_error(small_point, small_point_gt)])

            ef_list.append(np.abs(ef_pred.detach().cpu().numpy(
            ).squeeze() - ef_gt.detach().cpu().numpy().squeeze()))
            name_list.append(name)
            name_list.append(name)
            vef_pred_list.append(ef_pred.detach().cpu().numpy().squeeze())
            vef_gt_list.append(ef_gt.detach().cpu().numpy().squeeze())

            if args.viz:
                # save input
                # img = np.transpose(volume_batch.cpu().numpy().squeeze(), [
                #     1, 2, 0]) * 0.2 + 0.5
                img = np.transpose(volume_batch.cpu().numpy().squeeze(), [
                    1, 2, 0]) * 0.5 + 0.5
                small_img = img[:, :, :3]
                large_img = img[:, :, 3:]

                b, g, r = cv2.split(small_img)
                small_img_new1 = cv2.merge([r, g, b])
                b, g, r = cv2.split(large_img)
                large_img_new1 = cv2.merge([r, g, b])

                save_input_path = os.path.join(snapshot_path, 'viz/input_val/')
                if not os.path.exists(save_input_path):
                    os.makedirs(save_input_path)
                cv2.imwrite(save_input_path + name.replace('.h5',
                            '_small.png'), small_img_new1 * 255)
                cv2.imwrite(save_input_path + name.replace('.h5',
                            '_large.png'), large_img_new1 * 255)
                # save mask gt
                save_mask_gt_path = os.path.join(
                    snapshot_path, 'viz/gt/mask/')
                if not os.path.exists(save_mask_gt_path):
                    os.makedirs(save_mask_gt_path)
                cv2.imwrite(save_mask_gt_path + name.replace('.h5',
                            '_small.png'), small_map_gt * 255)
                cv2.imwrite(save_mask_gt_path + name.replace('.h5',
                            '_large.png'), large_map_gt * 255)
                # save boundary
                save_boundary_gt_path = os.path.join(
                    snapshot_path, 'viz/gt/boundary/')
                if not os.path.exists(save_boundary_gt_path):
                    os.makedirs(save_boundary_gt_path)

                cv2.imwrite(save_boundary_gt_path + name.replace('.h5',
                            '_small.png'), small_edge_gt * 255)
                cv2.imwrite(save_boundary_gt_path + name.replace('.h5',
                            '_large.png'), large_edge_gt * 255)
                # save point gt
                save_point_gt_path = os.path.join(
                    snapshot_path, 'viz/gt/point/')
                if not os.path.exists(save_point_gt_path):
                    os.makedirs(save_point_gt_path)
                plot_point(small_point_gt, small_map_gt.shape,
                           save_point_gt_path + name.replace('.h5', '_small.png'))
                plot_point(large_point_gt, large_map_gt.shape,
                           save_point_gt_path + name.replace('.h5', '_large.png'))

                # save prediction
                # save mask
                save_pred_mask_path = os.path.join(
                    snapshot_path, 'viz/pred/mask/')
                if not os.path.exists(save_pred_mask_path):
                    os.makedirs(save_pred_mask_path)

                cv2.imwrite(save_pred_mask_path + name.replace('.h5',
                            '_small.png'), small_map * 255)
                cv2.imwrite(save_pred_mask_path + name.replace('.h5',
                            '_large.png'), large_map * 255)
                # save boundary
                save_pred_boundary_path = os.path.join(
                    snapshot_path, 'viz/pred/boundary/')
                if not os.path.exists(save_pred_boundary_path):
                    os.makedirs(save_pred_boundary_path)

                cv2.imwrite(save_pred_boundary_path + name.replace('.h5',
                            '_small.png'), small_edge * 255)
                cv2.imwrite(save_pred_boundary_path + name.replace('.h5',
                            '_large.png'), large_edge * 255)
                # save point
                save_pred_point_path = os.path.join(
                    snapshot_path, 'viz/pred/point/')
                if not os.path.exists(save_pred_point_path):
                    os.makedirs(save_pred_point_path)
                plot_point(small_point, small_img_new1.shape,
                           save_pred_point_path + name.replace('.h5', '_small.png'))
                plot_point(large_point, large_img_new1.shape,
                           save_pred_point_path + name.replace('.h5', '_large.png'))

            if args.ras:
                # save ras
                save_ras_mask_path = os.path.join(
                    snapshot_path, 'viz/pred/ras/')
                if not os.path.exists(save_ras_mask_path):
                    os.makedirs(save_ras_mask_path)
                plot_ras_mask(small_point[1:-1, :], (256, 256),
                              save_ras_mask_path + name.replace('.h5', '_small.png'))
                plot_ras_mask(large_point[1:-1, :], (256, 256),
                              save_ras_mask_path + name.replace('.h5', '_large.png'))

            if args.tsne:
                # save tsne
                save_tsne_path = os.path.join(
                    snapshot_path, 'viz/tsne/')
                if not os.path.exists(save_tsne_path):
                    os.makedirs(save_tsne_path)
                pool.apply_async(plot_tsne, (small_proj, small_map_gt, small_edge_gt,
                                             save_tsne_path + name.replace('.h5', '_small.png')))
                pool.apply_async(plot_tsne, (large_proj, large_map_gt, large_edge_gt,
                                             save_tsne_path + name.replace('.h5', '_large.png')))
        if args.tsne:
            pool.close()
            pool.join()

        print('large_dice_mean:', np.array(large_list)[:, 0].mean())
        print('small_dice_mean:', np.array(small_list)[:, 0].mean())

        print('BC_large_mean:', np.array(large_list)[:, 2].mean())
        print('BC_small_mean:', np.array(small_list)[:, 2].mean())

        print('HD_large_mean:', np.array(large_list)[:, 3].mean())
        print('HD_small_mean:', np.array(small_list)[:, 3].mean())

        print('large_point_RMSE:', np.array(large_point_list)[:, 0].mean())
        print('small_point_RMSE:', np.array(small_point_list)[:, 0].mean())

        print('large_point_MAE:', np.array(large_point_list)[:, 1].mean())
        print('small_point_MAE:', np.array(small_point_list)[:, 1].mean())

        print('MAE_ef_mean:', mean_absolute_error(vef_pred_list, vef_gt_list))
        corr, _ = pearsonr(vef_pred_list, vef_gt_list)
        print('Person_ef_mean:', corr)

        if args.CI:
            # Confidence Interval#
            CI_large_dice = []
            CI_small_dice = []
            CI_vef = []
            CI_corr = []
            CI_HD_small = []
            CI_HD_large = []
            n_bootstraps = 2000
            rng_seed = 42  # control reproducibility
            rng = np.random.RandomState(rng_seed)
            for i in range(n_bootstraps):
                # bootstrap by sampling with replacement on the prediction indices
                indices = rng.randint(0, len(valloader), len(valloader))
                large_dice_CI = np.array(large_list)[indices, 0].mean()
                small_dice_CI = np.array(small_list)[indices, 0].mean()
                large_hd_CI = np.array(large_list)[indices, 3].mean()
                small_hd_CI = np.array(small_list)[indices, 3].mean()
                vef_CI = np.array(ef_list)[indices].mean()
                # print(np.array(ef_list)[indices])
                corr_CI, _ = pearsonr(np.array(vef_pred_list)[
                                      indices], np.array(vef_gt_list)[indices])
                # print(corr_CI[0])
                CI_large_dice.append(large_dice_CI)
                CI_small_dice.append(small_dice_CI)
                CI_vef.append(vef_CI)
                CI_corr.append(corr_CI)
                CI_HD_large.append(large_hd_CI)
                CI_HD_small.append(small_hd_CI)

            # large_dice_CI
            sorted_scores = np.sort(np.array(CI_large_dice))
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            print('95CI_large_dice, lower:, higher:',
                  confidence_lower, confidence_upper)
            # small_dice_CI
            sorted_scores = np.sort(np.array(CI_small_dice))
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            print('95CI_small_dice, lower:, higher:',
                  confidence_lower, confidence_upper)
            # large_hd_CI
            sorted_scores = np.sort(np.array(CI_HD_large))
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            print('95CI_large_hd95, lower:, higher:',
                  confidence_lower, confidence_upper)
            # small_hd_CI
            sorted_scores = np.sort(np.array(CI_HD_small))
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            print('95CI_small_hd95, lower:, higher:',
                  confidence_lower, confidence_upper)
            # vef_CI
            sorted_scores = np.sort(np.array(CI_vef))
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            print('95CI_vef, lower:, higher:',
                  confidence_lower, confidence_upper)
            # corr_CI
            sorted_scores = np.sort(np.array(CI_corr))
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            print('95CI_corr, lower:, higher:',
                  confidence_lower, confidence_upper)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = args.model_dir

    infer(args, snapshot_path)
