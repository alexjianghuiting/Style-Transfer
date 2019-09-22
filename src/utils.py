#coding=utf-8
#图像的读取和保存
import scipy.misc, numpy as np, os, sys

def save_img(out_path, img):
    # clip, limit the values in the array
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB') #(256, 256, 3)
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(path):
    files = []
    for (dirpath, dirnames, filesnames) in os.walk(path):
        files.extend(filesnames)
        break
    return [os.path.join(path, x) for x in files]