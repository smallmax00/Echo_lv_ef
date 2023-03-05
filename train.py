import argparse
import logging
import os
import glob
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from torch.optim.lr_scheduler import StepLR

from utils.utils import clip_gradient, saveModel
from lib.semiVBP import semiVBP


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../semiVBP_Datasets/Echonet', help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='h5py_point_norm_scaled',
                    help='Name of Dataset')
parser.add_argument('--exp', type=str,
                    default='EstAL-contrAD', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='semiVBP', help='model_name')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=26,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.06,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--clip', type=float,
                    default=0.5, help='gradient clipping margin')
parser.add_argument('--semi_epoch', type=int, default=20,
                    help='start epoch of semi-supervise')
parser.add_argument('--gamma', type=float, default=0.999,
                    help='gamma of stepLR')
parser.add_argument("-c", "--continue_training", type=str,
                    help="path of model if you want to continue a training")
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=13,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=835,
                    help='percent of labeled data')
# cross-validation
parser.add_argument("--fold", type=str, default='0')
# cuda id
parser.add_argument("--cuda_id", type=str,
                    default='0', help='cuda ID')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

# for unscaled
# 'point_avg': [[67.31532388, 55.4389139],
#                   [68.833462  , 57.53691141]],
# 'point_std': [[10.7595036 , 16.97739367],
#                 [14.15484566, 19.21685908]],

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
    'edge_contr_w': 0.01,  # 0.1 0.01
    'point_contr_w': 0.1,  # not used
    'ef_w': 0.01,  # 0.01
    'ul_w': 0.5,  # 0.5
    'semi_epoch': args.semi_epoch
}


def train(args, snapshot_path):

    base_lr = args.base_lr

    batch_size = args.batch_size
    max_epochs = args.max_epochs

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", dataset_name=args.dataset, fold=args.fold, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))

    db_val = BaseDataSets(base_dir=args.root_path, split="val",
                          dataset_name=args.dataset, fold=args.fold)

    total_slices = len(db_train)
    labeled_slice = args.labeled_num
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=10, shuffle=False,
                           num_workers=4)

    conf['max_epoch'] = max_epochs

    torch.cuda.empty_cache()

    model = semiVBP(conf)
    model = model.cuda()

    # smaller lr for point layer
    smallPoint_params = list(
        map(id, model.smallExtractor.point_layer.parameters()))
    largePoint_params = list(
        map(id, model.largeExtractor.point_layer.parameters()))
    point_params_id = smallPoint_params + largePoint_params
    base_params = filter(lambda p: id(
        p) not in point_params_id, model.parameters())
    point_params = filter(lambda p: id(
        p) in point_params_id, model.parameters())

    optimizer = optim.SGD([
        {'params': base_params},
        {'params': point_params, 'lr': base_lr / 6}],
        lr=base_lr, momentum=0.9, weight_decay=0.0001)

    scheduler = StepLR(optimizer, step_size=100, gamma=args.gamma)

    if args.continue_training:
        saved_model_path = args.continue_training
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        iter_num = checkpoint['epoch'] * len(trainloader) + 1
    else:
        start_epoch = 0
        iter_num = 0

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iterator = tqdm(range(start_epoch, conf['max_epoch']), ncols=70)

    small_best_dice = 0
    large_best_dice = 0
    ef_best_mse = 200

    for epoch_num in iterator:

        # train
        model.train()
        for i_batch, (sampled_batch, path) in enumerate(trainloader):

            input_batch, label_region, label_edge, label_point, label_ef = sampled_batch['image'], sampled_batch[
                'region'], sampled_batch['edge'], sampled_batch['point'], sampled_batch['ef']
            input_batch, label_region, label_edge, label_point, label_ef = input_batch.float().cuda(
            ), label_region.float().cuda(), label_edge.float().cuda(), label_point.float().cuda(), label_ef.float().cuda()

            if epoch_num > args.semi_epoch:
                loss, loss_list, small_loss, large_loss, ef_loss = model(input_batch[:args.labeled_bs],
                                                                         label_region[:args.labeled_bs], label_edge[:args.labeled_bs],
                                                                         label_point[:args.labeled_bs], label_ef[:args.labeled_bs],
                                                                         x_ul=input_batch[args.labeled_bs:],
                                                                         epoch=epoch_num, iter_num=iter_num)
            else:
                loss, loss_list, small_loss, large_loss, ef_loss = model(input_batch[:args.labeled_bs],
                                                                         label_region[:args.labeled_bs], label_edge[:args.labeled_bs],
                                                                         label_point[:args.labeled_bs], label_ef[:args.labeled_bs],
                                                                         epoch=epoch_num, iter_num=iter_num)

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            scheduler.step()

            iter_num = iter_num + 1
            writer.add_scalar(
                'info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/S_loss', small_loss, iter_num)
            writer.add_scalar('info/L_loss', large_loss, iter_num)
            writer.add_scalar('info/ef_loss', ef_loss, iter_num)
            writer.add_scalar('info/loss_region', loss_list[0], iter_num)
            writer.add_scalar('info/region_mid_loss', loss_list[1], iter_num)
            writer.add_scalar('info/loss_edge_dc', loss_list[2], iter_num)
            writer.add_scalar('info/loss_edge_ce', loss_list[3], iter_num)
            writer.add_scalar('info/loss_point', loss_list[4], iter_num)

            writer.add_scalar('info/edge_l_con', loss_list[5], iter_num)
            writer.add_scalar('info/point_l_con', loss_list[6], iter_num)
            writer.add_scalar('info/edge_l_contr', loss_list[7], iter_num)

            if epoch_num > args.semi_epoch:
                writer.add_scalar('info/edge_ul_con', loss_list[8], iter_num)
                writer.add_scalar('info/point_ul_con', loss_list[9], iter_num)
                writer.add_scalar('info/edge_ul_contr',
                                  loss_list[10], iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_region: %f, small_loss: %f, large_loss: %f' %
                (iter_num, loss.item(), loss_list[0].item(), small_loss.item(), large_loss.item()))

        if epoch_num % 50 == 49:
            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')

            saveModel(model, optimizer, scheduler, epoch_num, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # val
        if epoch_num % 2 == 0:
            model.eval()
            with torch.no_grad():

                small_total_dice = 0.0
                large_total_dice = 0.0
                ef_total_loss = 0.0
                for i_batch, (sampled_batch, path) in enumerate(valloader):

                    input_batch, label_region, label_edge, label_point, label_ef = sampled_batch['image'], sampled_batch[
                        'region'], sampled_batch['edge'], sampled_batch['point'], sampled_batch['ef']
                    input_batch, label_region, label_edge, label_point, label_ef = input_batch.float().cuda(
                    ), label_region.float().cuda(), label_edge.float().cuda(), label_point.float().cuda(), label_ef.float().cuda()

                    small_loss, large_loss, ef_loss, _, _, _, _, _, _, _, _, _, _, _ = model(
                        input_batch, label_region, label_edge, label_point, label_ef)

                    small_total_dice += 1 - small_loss
                    large_total_dice += 1 - large_loss
                    ef_total_loss += ef_loss
                small_avg_dice = small_total_dice / len(valloader)
                large_avg_dice = large_total_dice / len(valloader)
                ef_avg_mse = ef_total_loss / len(valloader)

                # save best models
                if small_avg_dice > small_best_dice:
                    small_best_dice = small_avg_dice
                    exist_small_model = glob.glob(
                        os.path.join(snapshot_path, "*small*"))
                    list(map(os.remove, exist_small_model))
                    save_small_model_path = os.path.join(
                        snapshot_path, 'epoch_{}_small_{}.pth'.format(
                            epoch_num, round(float(small_best_dice), 4))
                    )
                    saveModel(model, optimizer, scheduler,
                              epoch_num, save_small_model_path)
                    logging.info("save small best model to {}".format(
                        save_small_model_path))

                if large_avg_dice > large_best_dice:
                    large_best_dice = large_avg_dice
                    exist_large_model = glob.glob(
                        os.path.join(snapshot_path, "*large*"))
                    list(map(os.remove, exist_large_model))
                    save_large_model_path = os.path.join(
                        snapshot_path, 'epoch_{}_large_{}.pth'.format(
                            epoch_num, round(float(large_best_dice), 4))
                    )
                    saveModel(model, optimizer, scheduler,
                              epoch_num, save_large_model_path)
                    logging.info("save large best model to {}".format(
                        save_large_model_path))

                if ef_avg_mse < ef_best_mse:
                    ef_best_mse = ef_avg_mse
                    exist_ef_model = glob.glob(
                        os.path.join(snapshot_path, "*ef*"))
                    list(map(os.remove, exist_ef_model))
                    save_ef_model_path = os.path.join(
                        snapshot_path, 'epoch_{}_ef_{}.pth'.format(
                            epoch_num, round(float(ef_best_mse), 4))
                    )
                    saveModel(model, optimizer, scheduler,
                              epoch_num, save_ef_model_path)
                    logging.info("save ef best model to {}".format(
                        save_ef_model_path))

                writer.add_scalar('info/smallVal_dice',
                                  small_avg_dice, epoch_num)
                writer.add_scalar('info/largeVal_dice',
                                  large_avg_dice, epoch_num)
                writer.add_scalar('info/efVal_mse', ef_avg_mse, epoch_num)

    writer.close()
    return "Training Finished!"


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

    snapshot_path = "../semiVBP_Echo_model/{}/{}_labeled{}_b{}_lr{}_gamma{}_semi{}_edgeW{}_pointW{}_edgeCon{}_pointCon{}_edgeContr{}_efW{}_ulW{}_epoch{}/fold{}".format(
        args.exp, args.model, args.labeled_num, args.batch_size, args.base_lr, args.gamma, args.semi_epoch,
        conf['edge_w'], conf['point_w'], conf['edge_con_w'], conf['point_con_w'], conf['edge_contr_w'], conf['ef_w'], conf['ul_w'],
        args.max_epochs, args.fold)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
