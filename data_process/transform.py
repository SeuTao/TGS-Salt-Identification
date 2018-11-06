import cv2
import numpy as np
import os

def do_resize2(image, mask, H, W):
    image = cv2.resize(image,dsize=(W,H))
    mask = cv2.resize(mask,dsize=(W,H))
    mask  = (mask>0.5).astype(np.float32)

    return image,mask


#################################################################
def compute_center_pad(H,W, factor=32):
    if H%factor==0:
        dy0,dy1=0,0
    else:
        dy  = factor - H%factor
        dy0 = dy//2
        dy1 = dy - dy0

    if W%factor==0:
        dx0,dx1=0,0
    else:
        dx  = factor - W%factor
        dx0 = dx//2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


def do_center_pad_to_factor(image, factor=32):
    H,W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_center_pad(H,W, factor)

    image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REPLICATE)
    return image

def do_center_pad_to_factor_edgeYreflectX(image, factor=32):
    H,W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_center_pad(H,W, factor)

    image = cv2.copyMakeBorder(image, 0, 0, dx0, dx1, cv2.BORDER_REFLECT101)
    image = cv2.copyMakeBorder(image, dy0, dy1, 0, 0, cv2.BORDER_REPLICATE)
    return image


def do_center_pad_to_factor2(image, mask, factor=32):
    image = do_center_pad_to_factor(image, factor)
    mask  = do_center_pad_to_factor(mask, factor)
    return image, mask

#---
def do_horizontal_flip(image):
    #flip left-right
    image = cv2.flip(image,1)
    return image

def do_horizontal_flip2(image,mask):
    image = do_horizontal_flip(image)
    mask  = do_horizontal_flip(mask )
    return image, mask

#---

def compute_random_pad(H,W, limit=(-4,4), factor=32):
    if H%factor==0:
        dy0,dy1=0,0
    else:
        dy  = factor - H%factor
        dy0 = dy//2 + np.random.randint(limit[0],limit[1]) # np.random.choice(dy)
        dy1 = dy - dy0

    if W%factor==0:
        dx0,dx1=0,0
    else:
        dx  = factor - W%factor
        dx0 = dx//2 + np.random.randint(limit[0],limit[1]) # np.random.choice(dx)
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


def do_random_pad_to_factor2(image, mask, limit=(-4,4), factor=32):
    H,W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_random_pad(H,W, limit, factor)

    image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REPLICATE)
    mask  = cv2.copyMakeBorder(mask,  dy0, dy1, dx0, dx1, cv2.BORDER_REPLICATE)

    return image, mask

def do_random_pad_to_factor2_edgeYreflectX(image, mask, limit=(-4,4), factor=32):
    H,W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_random_pad(H,W, limit, factor)

    # image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REPLICATE)
    image = cv2.copyMakeBorder(image, 0, 0, dx0, dx1, cv2.BORDER_REFLECT101)
    image = cv2.copyMakeBorder(image, dy0, dy1, 0, 0, cv2.BORDER_REPLICATE)

    # mask  = cv2.copyMakeBorder(mask,  dy0, dy1, dx0, dx1, cv2.BORDER_REPLICATE)
    mask = cv2.copyMakeBorder(mask, 0, 0, dx0, dx1, cv2.BORDER_REFLECT101)
    mask = cv2.copyMakeBorder(mask, dy0, dy1, 0, 0, cv2.BORDER_REPLICATE)

    return image, mask

#----
def do_invert_intensity(image):
    #flip left-right
    image = np.clip(1-image,0,1)
    return image


def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image


def do_brightness_multiply(image, alpha=1):
    image = alpha*image
    image = np.clip(image, 0, 1)
    return image


#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):

    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image


def do_flip_transpose2(image, mask, type=0):
    #choose one of the 8 cases

    if type==1: #rotate90
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)


    if type==2: #rotate180
        image = cv2.flip(image,-1)
        mask  = cv2.flip(mask,-1)


    if type==3: #rotate270
        image = image.transpose(1,0)
        image = cv2.flip(image,0)

        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,0)


    if type==4: #flip left-right
        image = cv2.flip(image,1)
        mask  = cv2.flip(mask,1)


    if type==5: #flip up-down
        image = cv2.flip(image,0)
        mask  = cv2.flip(mask,0)

    if type==6:
        image = cv2.flip(image,1)
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

        mask = cv2.flip(mask,1)
        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)

    if type==7:
        image = cv2.flip(image,0)
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

        mask = cv2.flip(mask,0)
        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)


    return image, mask

##================================
def do_shift_scale_crop( image, mask, x0=0, y0=0, x1=1, y1=1 ):
    #cv2.BORDER_REFLECT_101
    #cv2.BORDER_CONSTANT
    height, width = image.shape[:2]
    image = image[y0:y1,x0:x1]
    mask  = mask [y0:y1,x0:x1]

    image = cv2.resize(image,dsize=(width,height))
    mask  = cv2.resize(mask,dsize=(width,height))
    mask  = (mask>0.5).astype(np.float32)
    return image, mask


def do_random_shift_scale_crop_pad2(image, mask, limit=0.10):
    H, W = image.shape[:2]

    dy = int(H*limit)
    y0 =   np.random.randint(0,dy)
    y1 = H-np.random.randint(0,dy)

    dx = int(W*limit)
    x0 =   np.random.randint(0,dx)
    x1 = W-np.random.randint(0,dx)

    #y0, y1, x0, x1
    image, mask = do_shift_scale_crop( image, mask, x0, y0, x1, y1 )
    return image, mask

#===========================================================================

def resize_and_pad(image, resize_size, factor):
    image = cv2.resize(image, (resize_size,resize_size))
    image = do_center_pad_to_factor(image, factor)
    return image

def resize_and_pad_edgeYreflectX(image, resize_size, factor):
    image = cv2.resize(image, (resize_size,resize_size))
    image = do_center_pad_to_factor_edgeYreflectX(image, factor)
    return image

def resize_and_random_pad(image, mask, resize_size, factor,limit=(-13, 13)):
    image = cv2.resize(image, (resize_size,resize_size))
    mask = cv2.resize(mask, (resize_size, resize_size))
    image, mask = do_random_pad_to_factor2(image, mask, limit = limit, factor = factor)
    return image, mask

def resize_and_random_pad_edgeYreflectX(image, mask, resize_size, factor):
    image = cv2.resize(image, (resize_size,resize_size))
    mask = cv2.resize(mask, (resize_size, resize_size))

    image, mask = do_random_pad_to_factor2_edgeYreflectX(image, mask, limit=(-13, 13), factor = factor)
    return image, mask

def center_corp(image, image_size, crop_size):
    image = cv2.resize(image, (image_size, image_size))
    radius = (image_size - crop_size)/2
    image = image[radius:radius+crop_size,radius:radius+crop_size]
    return image


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

def save_csv_images(csv_path, save_path):
    dict = decode_csv(csv_name=csv_path)

    for id in dict:
        id_img = dict[id]*255
        cv2.imwrite(os.path.join(save_path,id+'.png'),id_img)
