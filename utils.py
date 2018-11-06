import logging
import os
# import pathlib
import random
import sys
import time
from itertools import chain
from collections import Iterable

# from deepsense import neptune
import numpy as np
import pandas as pd
import torch
from PIL import Image
import yaml
from imgaug import augmenters as iaa
import imgaug as ia

# NEPTUNE_CONFIG_PATH = str(pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'neptune.yaml')

import argparse
from utils import *
import numpy as np
import os
import glob
import cv2
from jigsaw.handcraft_ruls_postprocessing import submission_apply_jigsaw_postprocessing

def save_csv_images(csv_path, save_path):
    dict = decode_csv(csv_name=csv_path)

    for id in dict:
        id_img = dict[id]*255
        cv2.imwrite(os.path.join(save_path,id+'.png'),id_img)

def create_csv_lists(image_dir, printable = True):
    if not os.path.exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    file_list = []
    file_glob = os.path.join(image_dir,'*.' + 'csv')
    file_list.extend(glob.glob(file_glob))
    if printable:
        print(len(file_list))
    return file_list

def create_csv_lists_recursive(image_dir):
    total_list = []
    for i in os.walk(image_dir):
        cur_path = i[0]
        list = create_csv_lists(cur_path,printable=False)
        total_list.extend(list)

    print(len(total_list))
    return total_list

def do_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b>prev+1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    #if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
    #    rle[-2] = rle[-2] -1

    rle = ' '.join([str(r) for r in rle])
    return rle

from math import isnan
def do_length_decode(rle, H, W, fill_value=255):
    mask = np.zeros((H,W), np.uint8)
    if type(rle).__name__ == 'float': return mask

    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask

def decode_csv(csv_name):
    import pandas as pd
    data = pd.read_csv(csv_name)
    id = data['id']
    rle_mask = data['rle_mask']

    dict = {}
    for id, rle in zip(id,rle_mask):
        tmp = do_length_decode(rle, 101, 101, fill_value=1)
        dict[id] = tmp

    return dict

def save_id_fea(predict_dict, save_dir):
    for id in predict_dict:
        output_mat = predict_dict[id].astype(np.float32)
        np.save(os.path.join(save_dir,id), output_mat)

def state_dict_remove_moudle(moudle_state_dict, model):
    state_dict = model.state_dict()
    keys = list(moudle_state_dict.keys())
    for key in keys:
        print(key + ' loaded')
        new_key = key.replace(r'module.', r'')
        print(new_key)
        state_dict[new_key] = moudle_state_dict[key]

    return state_dict

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks

def encode_rle(predictions):
    return [run_length_encoding(mask) for mask in predictions]

def create_submission(predictions):
    output = []
    for image_id, mask in predictions:
        # print(image_id)

        rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask))
        output.append([image_id, rle_encoded])

    submission = pd.DataFrame(output, columns=['id', 'rle_mask']).astype(str)
    return submission

def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] > (x.size+1):
        rle[-2] = rle[-2] - 1

    return rle

def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T

def get_crop_pad_sequence(vertical, horizontal):
    top = int(vertical / 2)
    bottom = vertical - top
    right = int(horizontal / 2)
    left = horizontal - right
    return (top, right, bottom, left)

def get_list_of_image_predictions(batch_predictions):
    image_predictions = []
    for batch_pred in batch_predictions:
        image_predictions.extend(list(batch_pred))
    return image_predictions

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed

def reseed(augmenter, deterministic=True):
    augmenter.random_state = ia.new_random_state(get_seed())
    if deterministic:
        augmenter.deterministic = True

    for lists in augmenter.get_children_lists():
        for aug in lists:
            aug = reseed(aug, deterministic=True)
    return augmenter
