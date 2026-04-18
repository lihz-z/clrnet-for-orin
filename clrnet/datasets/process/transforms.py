import random
import cv2
import numpy as np
import torch
import numbers
import collections
from PIL import Image

from ..registry import PROCESS


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PROCESS.register_module
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys=['img', 'mask'], cfg=None):
        self.keys = keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in self.keys:
            if key == 'img_metas' or key == 'gt_masks' or key == 'lane_line':
                data[key] = sample[key]
                continue
            data[key] = to_tensor(sample[key])
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PROCESS.register_module
class RandomLROffsetLABEL(object):
    def __init__(self, max_offset, cfg=None):
        self.max_offset = max_offset

    def __call__(self, sample):
        img = sample['img']
        label = sample['mask']
        offset = np.random.randint(-self.max_offset, self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[:, offset:, :] = img[:, 0:w - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:
            real_offset = -offset
            img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
            img[:, w - real_offset:, :] = 0

        label = np.array(label)
        if offset > 0:
            label[:, offset:] = label[:, 0:w - offset]
            label[:, :offset] = 0
        if offset < 0:
            offset = -offset
            label[:, 0:w - offset] = label[:, offset:]
            label[:, w - offset:] = 0
        sample['img'] = img
        sample['mask'] = label

        return sample


@PROCESS.register_module
class RandomUDoffsetLABEL(object):
    def __init__(self, max_offset, cfg=None):
        self.max_offset = max_offset

    def __call__(self, sample):
        img = sample['img']
        label = sample['mask']
        offset = np.random.randint(-self.max_offset, self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        sample['img'] = img
        sample['mask'] = label
        return sample


@PROCESS.register_module
class Resize(object):
    def __init__(self, size, cfg=None):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        out = list()
        sample['img'] = cv2.resize(sample['img'],
                                   self.size,
                                   interpolation=cv2.INTER_CUBIC)
        if 'mask' in sample:
            sample['mask'] = cv2.resize(sample['mask'],
                                        self.size,
                                        interpolation=cv2.INTER_NEAREST)
        return sample


@PROCESS.register_module
class RandomCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images


@PROCESS.register_module
class CenterCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = max(0, int((h - th) / 2))
        w1 = max(0, int((w - tw) / 2))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images


@PROCESS.register_module
class RandomRotation(object):
    def __init__(self,
                 degree=(-10, 10),
                 interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST),
                 padding=None,
                 cfg=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding
        if self.padding is None:
            self.padding = [0, 0]

    def _rotate_img(self, sample, map_matrix):
        h, w = sample['img'].shape[0:2]
        sample['img'] = cv2.warpAffine(sample['img'],
                                       map_matrix, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=self.padding)

    def _rotate_mask(self, sample, map_matrix):
        if 'mask' not in sample:
            return
        h, w = sample['mask'].shape[0:2]
        sample['mask'] = cv2.warpAffine(sample['mask'],
                                        map_matrix, (w, h),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=self.padding)

    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = sample['img'].shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            self._rotate_img(sample, map_matrix)
            self._rotate_mask(sample, map_matrix)
        return sample


@PROCESS.register_module
class RandomBlur(object):
    def __init__(self, applied, cfg=None):
        self.applied = applied

    def __call__(self, img_group):
        assert (len(self.applied) == len(img_group))
        v = random.random()
        if v < 0.5:
            out_images = []
            for img, a in zip(img_group, self.applied):
                if a:
                    img = cv2.GaussianBlur(img, (5, 5),
                                           random.uniform(1e-6, 0.6))
                out_images.append(img)
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][
                        ..., np.newaxis]  # single channel image
            return out_images
        else:
            return img_group


@PROCESS.register_module
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy Image with a probability of 0.5
    """
    def __init__(self, cfg=None):
        pass

    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            sample['img'] = np.fliplr(sample['img'])
            if 'mask' in sample: sample['mask'] = np.fliplr(sample['mask'])
        return sample


@PROCESS.register_module
class Normalize(object):
    def __init__(self, img_norm, cfg=None):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        m = self.mean
        s = self.std
        img = sample['img']
        if len(m) == 1:
            img = img - np.array(m)  # single channel image
            img = img / np.array(s)
        else:
            img = img - np.array(m)[np.newaxis, np.newaxis, ...]
            img = img / np.array(s)[np.newaxis, np.newaxis, ...]
        sample['img'] = img

        return sample


@PROCESS.register_module
class RainRobustAug(object):
    """Image-only augmentations tailored for rainy lane scenes."""

    def __init__(self,
                 p=0.8,
                 reflection_p=0.35,
                 occlusion_p=0.25,
                 contrast_p=0.5,
                 cfg=None):
        self.p = p
        self.reflection_p = reflection_p
        self.occlusion_p = occlusion_p
        self.contrast_p = contrast_p

    def _apply_contrast(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        gamma = random.uniform(0.75, 1.35)
        v = np.power(v.astype(np.float32) / 255.0, gamma) * 255.0
        hsv[:, :, 2] = np.clip(v, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if random.random() < 0.5:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.0),
                                    tileGridSize=(8, 8))
            lab = cv2.merge((clahe.apply(l), a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img

    def _add_reflection(self, img):
        h, w = img.shape[:2]
        overlay = np.zeros_like(img, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        num_regions = random.randint(1, 4)

        for _ in range(num_regions):
            center = (random.randint(0, w - 1), random.randint(h // 2, h - 1))
            axes = (random.randint(w // 20, w // 8),
                    random.randint(h // 30, h // 12))
            angle = random.randint(0, 180)
            strength = random.uniform(35, 90)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
            cv2.ellipse(overlay, center, axes, angle, 0, 360,
                        (strength, strength, strength), -1)

        blur_size = random.choice([31, 41, 51])
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
        overlay *= mask[..., None] * random.uniform(0.35, 0.6)
        img = np.clip(img.astype(np.float32) + overlay, 0, 255)
        return img.astype(np.uint8)

    def _add_occlusion(self, img):
        h, w = img.shape[:2]
        num_blocks = random.randint(1, 3)
        for _ in range(num_blocks):
            block_w = random.randint(max(20, w // 20), max(40, w // 8))
            block_h = random.randint(max(20, h // 20), max(40, h // 8))
            x1 = random.randint(0, max(0, w - block_w))
            y1 = random.randint(h // 3, max(h // 3, h - block_h))
            patch = img[y1:y1 + block_h, x1:x1 + block_w]
            fill = patch.mean(axis=(0, 1), keepdims=True)
            img[y1:y1 + block_h, x1:x1 + block_w] = fill.astype(np.uint8)
        return img

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample

        img = sample['img'].astype(np.uint8)
        if random.random() < self.contrast_p:
            img = self._apply_contrast(img)
        if random.random() < self.reflection_p:
            img = self._add_reflection(img)
        if random.random() < self.occlusion_p:
            img = self._add_occlusion(img)

        sample['img'] = img
        return sample


def CLRTransforms(img_h, img_w):
    return [
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
        dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
        dict(name='Affine',
             parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                    y=(-0.1, 0.1)),
                             rotate=(-10, 10),
                             scale=(0.8, 1.2)),
             p=0.7),
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
    ]
