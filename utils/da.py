import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from cvtransforms import transforms
from cvtransforms import functional as tf
import datetime

class Augmentations_PIL:
    def __init__(self, augment_mode,h,w):
        self.augment_dict=augment_mode
        self.image_fill = 0
        self.label_fill = 0
        self.input_hw=[h,w]

    def rotation(self, image, label):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :param angle:  None, list-float, tuple-float
        :return:  PIL
        '''
        angle1 = self.augment_dict["rotation"]["degrees"]
        # if angle is None:
        angle = transforms.RandomRotation.get_params(angle1)
        # elif isinstance(angle, list) or isinstance(angle, tuple):
        #     angle = random.choice(angle)
        image = tf.rotate(image, angle, fill=self.image_fill)
        label = tf.rotate(label, angle, fill=self.label_fill)
        return image, label

    def flipH(self, image, label):
        image = tf.hflip(image)
        label = tf.hflip(label)
        return image, label

    def flipV(self,image,label):
        image = tf.vflip(image)
        label = tf.vflip(label)
        return image,label

    def resizecrop(self, image, label):

        size ,scale = self.augment_dict["resizecrop"]["size"], self.augment_dict["resizecrop"]["scale"]
        image=transforms.RandomResizedCrop(size,(scale,1.0))(image)
        label=transforms.RandomResizedCrop(size,(scale,1.0))(label)

        return image, label

    def perspective(self, image, label):
        distortion_scale = self.augment_dict["perspective"]["distortion_scale"]

        width, height = image.size
        startpoints, endpoints = transforms.RandomPerspective.get_params(width, height, distortion_scale)
        image = tf.perspective(image, startpoints, endpoints, interpolation=Image.BICUBIC, fill=self.image_fill)
        label = tf.perspective(label, startpoints, endpoints, interpolation=Image.NEAREST, fill=self.label_fill)

        return image, label

    def affine(self, image, label):
        degrees, translate, scale_ranges = self.augment_dict["affine"]["degrees"],self.augment_dict["affine"]["translate"],self.augment_dict["affine"]["scale"]
        shears=self.augment_dict["affine"]["shear"]
        ret = transforms.RandomAffine.get_params(degrees, translate, scale_ranges,shears, img_size=image.size)
        image = tf.affine(image, *ret, resample=0, fillcolor=self.image_fill)  # PIL.Image.NEAREST
        label = tf.affine(label, *ret, resample=0, fillcolor=self.label_fill)

        return image, label

    def colorjitter(self, image, label):
        brightness,contrast= self.augment_dict["colorjitter"]["brightness"], self.augment_dict["colorjitter"]["contrast"]
        saturation ,hue=self.augment_dict["colorjitter"]["saturation"], self.augment_dict["colorjitter"]["hue"]
        transforms_func = transforms.ColorJitter(brightness=brightness,
                                                 contrast=contrast,
                                                 saturation=saturation,
                                                 hue=hue)
        image = transforms_func(image)
        return image, label

    # gassian noise
    def gaussiannoise(self, image, label):
        mean ,std = self.augment_dict["gaussiannoise"]["mean"], self.augment_dict["gaussiannoise"]["std"]
        transforms_func = transforms.RandomGaussianNoise(0.9, mean, std)
        image = transforms_func(image)

        return image, label

    def spnoise(self, image, label):
        prob=self.augment_dict["spnoise"]["prob"]
        transforms_func = transforms.RandomSPNoise(0.9, prob)
        image = transforms_func(image)
        return image, label
    def self_images(self,image,label):
        return image, label


class Transforms_PIL(object):
    def __init__(self, augment_list):
        self.aug_pil = Augmentations_PIL(augment_list)
        self.augment_ways=augment_list
        self.aug_funcs = [a for a in self.aug_pil.__dir__() if not a.startswith('_') and a not in self.aug_pil.__dict__]

    def __call__(self, image, label):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :return:  PIL
        '''
        if len(self.augment_ways.keys())==0:
            return -1
        elif len(self.augment_ways.keys())==1:
            name=list(self.augment_ways.keys())[0] #取出来并转成字符串的形式
            image, label = getattr(self.aug_pil, str(name))(image, label)
            return image, label
        else:
            name=np.random.choice(list(self.augment_ways.keys()))
            image, label = getattr(self.aug_pil, str(name))(image, label)
            return image, label


class ToTensor(object):
    # image label -> tensor, image div 255
    def __call__(self, image, label):
        # PIL uint8
        image = tf.to_tensor(image)  # transpose HWC->CHW, /255
        label = torch.from_numpy(np.array(label))  # PIL->ndarray->tensor
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label


def get_current_time():
    current_time = datetime.datetime.now()
    year = str(current_time.year)
    month = str(current_time.month)
    day = str(current_time.day)
    hour = str(current_time.hour)
    minute = str(current_time.minute)
    second = str(current_time.second)
    microsecond = str(current_time.microsecond)
    current_time_str = year + month + day + hour + minute + second + "_" + microsecond
    return current_time_str

