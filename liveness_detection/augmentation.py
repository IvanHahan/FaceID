# import imgaug.augmenters as iaa
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from utils.image_processing import resize_image, pad_image
import cv2


# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


# def boxes_numpy2imgaug(annotations, image_shape):
#     boxes = []
#     for annot in annotations:
#         x1, y1, x2, y2, label = annot
#         box = BoundingBox(x1, y1, x2, y2, label)
#         boxes.append(box)
#     return BoundingBoxesOnImage(boxes, image_shape)
#
#
# def boxes_imgaug2numpy(boxes):
#     annotations = []
#     for box in boxes.bounding_boxes:
#         annotations.append((box.x1, box.y1, box.x2, box.y2, box.label))
#     return np.array(annotations)
#
#
# def get_augmentations():
#     def sometimes(aug, p=0.5): return iaa.Sometimes(p, aug)
#
#     return iaa.Sequential(
#         [
#             iaa.SomeOf(2, [
#                 sometimes(iaa.Multiply()),
#                 sometimes(iaa.HorizontalFlip()),
#                 sometimes(iaa.GammaContrast()),
#                 sometimes(iaa.AddToHueAndSaturation(5)),
#                 sometimes(iaa.CLAHE())
#             ]
#                        )
#         ]
#     )


# class Augmenter(object):
#     def __init__(self, augmentations=get_augmentations()):
#         self.augmentations = augmentations
#
#     def __call__(self, sample):
#         """
#         :param image: numpy image
#         :param annotations: (x1, x2, y1, y2, label) unscaled
#         """
#         image = sample['image']
#         if 'annotations' in sample:
#             annotations = sample['annotations']
#             boxes = boxes_numpy2imgaug(annotations, image.shape[:2])
#             image_aug, boxes_aug = self.augmentations(image=image, bounding_boxes=boxes)
#             sample['image'] = image_aug
#             sample['annotations'] = boxes_imgaug2numpy(boxes_aug)
#         else:
#             image_aug = self.augmentations(image=image)
#             sample['image'] = image_aug
#
#         return sample


class MaxSizeResizer(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        """
        :param image: numpy image
        :param annotations: (x1, x2, y1, y2, label) unscaled
        """
        if isinstance(sample, dict):
            image = sample['image']
            if 'annotations' in sample:
                annotations = sample['annotations']
                d = max(image.shape) / self.size
                annotations[:, :4] /= d
                annotations = annotations.astype(int)
                sample['annotations'] = annotations

            image = resize_image(image, self.size)
            sample['image'] = image
        else:
            sample = resize_image(sample, self.size)

        return sample


class SquarePad(object):

    def __call__(self, sample):
        """
        :param image: numpy image
        :param annotations: (x1, x2, y1, y2, label) unscaled
        """
        if isinstance(sample, dict):
            image = sample['image']

            image = pad_image(image, (max(image.shape), max(image.shape)))[0]
            sample['image'] = image
        else:
            sample = pad_image(sample, (max(sample.shape), max(sample.shape)))[0]

        return sample


class ToTensor(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['image'] = torch.from_numpy(sample['image'].transpose([2, 0, 1]))
            if 'annotations' in sample:
                sample['annotations'] = torch.from_numpy(sample['annotations'])
        else:
            sample = torch.from_numpy(sample.transpose([2, 0, 1])).double()
        return sample


class Preprocessor(object):

    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, samples):
        """samples: pil images"""

        transformations = transforms.Compose([
            MaxSizeResizer(224),
            SquarePad(),
            transforms.ToPILImage(),
        ])
        samples = [transformations(s) for s in samples]

        if self.augment:
            rotation, translation, scale, shear = transforms.RandomAffine.get_params((-30, 30), (0.1, 0.1),
                                                                                     (0.8, 1.2), None,
                                                                                     samples[0].size)
            brightness = np.random.uniform(0.5, 1.5)
            contrast = np.random.uniform(0.7, 1.3)
            hue = np.random.uniform(-0.2, 0.2)
            flip = np.random.uniform(0, 1) > 0.5

            for i, s in enumerate(samples):
                s = F.affine(s, rotation, translation, scale, shear)
                s = F.adjust_brightness(s, brightness)
                s = F.adjust_contrast(s, contrast)
                s = F.adjust_hue(s, hue)
                if flip > 0.5:
                    s = F.hflip(s)
                samples[i] = s
            # plt.imshow(np.array(samples[0]))
            # plt.show()

        transformations = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.456,
                                 std=0.224),
        ])
        samples = torch.cat([transformations(s) for s in samples], 0)
        return samples


class ToFlow(object):

    def __call__(self, images):
        flows = []
        resize = MaxSizeResizer(224)

        prev_image = images[0]
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        for image in images[1:]:
            prev_image = resize(prev_image)
            image = cv2.resize(image, (prev_image.shape[1], prev_image.shape[0]))

            hsv = np.zeros((*prev_image.shape, 3), dtype='uint8')

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # plt.imshow(rgb)
            # plt.show()
            flows.append(rgb)
            prev_image = image
        return flows
