from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from torchvision.transforms import functional


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', dataset_name=None, fold=None, testlist=None, num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.dataset_name = dataset_name
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/cv/train.'+fold+'.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/cv/val.'+fold+'.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'test':
            with open(testlist, 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'infer':
            with open(self._base_dir + '/infer.ad.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(join(self._base_dir, self.dataset_name, case), 'r')
        image = h5f['img'][:]
        region = h5f['region'][:]  # * 255
        edge = h5f['edge'][:]
        point = h5f['point'][:]
        ef = np.array(h5f['AL_ef'])

        if self.split == "train":
            sample = {'image': image, 'region': region,
                      'edge': edge, 'point': point, 'ef': ef}
            sample = self.transform(sample)
            sample["idx"] = idx
        else:
            image = functional.to_tensor(image.astype(np.float32))
            # for Echonet
            small_img, large_img = image[:3], image[3:]
            small_img, large_img = functional.rgb_to_grayscale(
                small_img, 3), functional.rgb_to_grayscale(large_img, 3)
            small_img = functional.normalize(
                small_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            large_img = functional.normalize(
                large_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            image = torch.cat((small_img, large_img), dim=0)

            image, _ = pad_to(image, 32)
            region = functional.to_tensor(region.astype(np.float32))
            edge = functional.to_tensor(edge.astype(np.float32))
            point = functional.to_tensor(point.astype(np.float32))
            ef = torch.tensor(ef.astype(np.float32))

            sample = {'image': image, 'region': region,
                      'edge': edge, 'point': point, 'ef': ef}
        return sample, case


class inferDataSets(Dataset):
    def __init__(self, base_dir=None, dataset_name=None, inferlist=None) -> None:
        super().__init__()
        self._base_dir = base_dir
        self.sample_list = []
        self.dataset_name = dataset_name
        with open(inferlist, 'r') as f:
            self.sample_list = f.readlines()
        self.sample_list = [item.replace('\n', '')
                            for item in self.sample_list]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(join(self._base_dir, self.dataset_name, case), 'r')
        image = h5f['img'][:]

        image = functional.to_tensor(image.astype(np.float32))
        small_img, large_img = image[:3], image[3:]
        small_img, large_img = functional.rgb_to_grayscale(
            small_img, 3), functional.rgb_to_grayscale(large_img, 3)
        small_img = functional.normalize(
            small_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        large_img = functional.normalize(
            large_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image = torch.cat((small_img, large_img), dim=0)
        image, _ = pad_to(image, 32)

        return {'image': image}, case


def rotCoord(coords, degree, size):
    degree = np.deg2rad(degree)
    center = np.array([size[0] // 2, size[1] // 2], dtype=float)[None, ...]
    rotM = np.array([[np.cos(degree), -np.sin(degree)],
                     [np.sin(degree), np.cos(degree)]], dtype=float)
    rotated = (coords - center) @ rotM + center
    return rotated


def flipCoord(coords, axis, size):
    flipped = coords.copy()
    if axis == 0:
        flipped[:, 1] = size[0] - flipped[:, 1]
    elif axis == 1:
        flipped[:, 0] = size[1] - flipped[:, 0]

    return flipped


def random_rot_flip(item_list):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    for i in range(len(item_list) - 1):
        item_list[i] = np.rot90(item_list[i], k)
        item_list[i] = np.flip(item_list[i], axis=axis).copy()
    # point
    size = item_list[0].shape
    item_list[-1] = rotCoord(item_list[-1], k * 90, size)
    item_list[-1] = flipCoord(item_list[-1], axis, size)

    return item_list


def random_rotate(item_list):
    angle = np.random.randint(-20, 20)
    for i in range(len(item_list) - 1):
        item_list[i] = ndimage.rotate(
            item_list[i], angle, order=0, reshape=False)
    # point
    size = item_list[0].shape
    item_list[-1] = rotCoord(item_list[-1], angle, size)

    return item_list


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        item_list = [sample['image'], sample['region'], sample['edge'],
                     sample['point'], sample['ef']]  # , sample['sdm']
        image, region, edge, point, ef = tuple(item_list)

        image = functional.to_tensor(image.astype(np.float32))
        # for Echonet
        small_img, large_img = image[:3], image[3:]
        small_img, large_img = functional.rgb_to_grayscale(
            small_img, 3), functional.rgb_to_grayscale(large_img, 3)
        small_img = functional.normalize(
            small_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        large_img = functional.normalize(
            large_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image = torch.cat((small_img, large_img), dim=0)

        image, _ = pad_to(image, 32)
        region = functional.to_tensor(region.astype(np.float32))
        edge = functional.to_tensor(edge.astype(np.float32))
        point = functional.to_tensor(point.astype(np.float32))
        ef = torch.tensor(ef.astype(np.float32))

        sample = {'image': image, 'region': region,
                  'edge': edge, 'point': point, 'ef': ef}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
