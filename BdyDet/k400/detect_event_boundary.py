from glob import glob
import os.path as osp
import os
import pickle
import re
import shutil
import subprocess
import sys
from time import time
import tempfile
import json

import cv2
import ipdb
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters
from scipy import ndimage
import tqdm

import io
import base64
from IPython.display import HTML

def sizeof_fmt(num, suffix="B"):
    """
    Returns the filesize as human readable string.
    https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-
        readable-version-of-file-size
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def get_dir_size(dirname):
    """
    Returns the size of the contents of a directory. (Doesn't include subdirs.)
    """
    size = 0
    for fname in os.listdir(dirname):
        fname = os.path.join(dirname, fname)
        if os.path.isfile(fname):
            size += os.path.getsize(fname)
    return size


class VideoWriter(object):
    def __init__(self, output_path, fps, temp_dir=None):
        self.output_path = output_path
        self.fps = fps
        self.temp_dir = temp_dir
        self.current_index = 0
        self.img_shape = None
        self.frame_string = "frame{:08}.png"

    def add_images(self, images_list, show_pbar=False):
        """
        Adds a list of images to temporary directory.
        Args:
            images_list (iterable): List of images (HxWx3).
            show_pbar (bool): If True, displays a progress bar.
        Returns:
            list: filenames of saved images.
        """
        filenames = []
        if show_pbar:
            images_list = tqdm(images_list)
        for image in images_list:
            filenames.append(self.add_image(image))
        return filenames

    def add_image(self, image):
        """
        Saves image to file.
        Args:
            image (HxWx3).
        Returns:
            str: filename.
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        if self.img_shape is None:
            self.img_shape = image.shape
        assert self.img_shape == image.shape
        filename = self.get_filename(self.current_index)
        plt.imsave(fname=filename, arr=image)
        self.current_index += 1
        return filename

    def get_frame(self, index):
        """
        Read image from file.
        Args:
            index (int).
        Returns:
            Array (HxWx3).
        """
        filename = self.get_filename(index)
        return plt.imread(fname=filename)

    def get_filename(self, index):
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        return os.path.join(self.temp_dir, self.frame_string.format(index))

    def make_video(self):
        cmd = (
            "ffmpeg -y -threads 16 -framerate {fps} "
            "-i {temp_dir}/frame%08d.png -profile:v baseline -level 3.0 "
            "-c:v libx264 -pix_fmt yuv420p -an -vf "
            '"scale=trunc(iw/2)*2:trunc(ih/2)*2" {output_path}'.format(
                fps=self.fps, temp_dir=self.temp_dir, output_path=self.output_path
            )
        )
        print(cmd)
        try:
            subprocess.call(cmd, shell=True)
        except OSError:
            ipdb.set_trace()
            print("OSError")

    def close(self):
        """
        Clears the temp_dir.
        """
        print(
            "Removing {} which contains {}.".format(
                self.temp_dir, self.get_temp_dir_size()
            )
        )
        shutil.rmtree(self.temp_dir)
        self.temp_dir = None

    def get_temp_dir_size(self):
        """
        Returns the size of the temp dir.
        """
        return sizeof_fmt(get_dir_size(self.temp_dir))
    

def main(vid_id):
    
    print("process video: " + str(vid_id))
    
    # deal with index/position
    offset = 4 # this is after downsample, the end frame index of the current segment (4-th) of the first window
    downsample = 3
    offset *= downsample
    
    # hyper-para for boundary detection
    smooth_factor = 5
    LoG_sigma = 15

    # Update these
    exp_path = '../../data/exp_k400/'
    output_seg_dir = 'detect_seg'
    SAVE_PATH = exp_path + output_seg_dir + '/{}/frame_{{:05d}}.jpg'
    OUTPUT_PATH = exp_path + output_seg_dir + '/{}.mp4'
    OUTPUT_BDY_PATH = exp_path + output_seg_dir + '/{}.pkl'

    if not os.path.exists(exp_path + output_seg_dir):
        os.makedirs(exp_path + output_seg_dir)

    def detect_boundary(signal, offset, sigma):
        bdy_idx = []        
        LoG_mean_errors = ndimage.gaussian_laplace(signal, sigma=LoG_sigma, mode='nearest')
        delta_LoG = np.gradient(LoG_mean_errors)
        for i in range(len(signal)-1):
            if delta_LoG[i]>=0 and delta_LoG[i+1]<=0:
                if delta_LoG[i] > -delta_LoG[i+1]:
                    bdy_idx += [i+offset+1]
                else:
                    bdy_idx += [i+offset]
        return bdy_idx

    # read and pre-process prediction error for each video
    windows = sorted(glob(exp_path + 'pred_err/err_vid_' + vid_id + '*.pkl'))
    if windows == []:
        return
        
    pred_err = []
    for window_path in windows:
        with open(window_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            pred_err.append(data['pred_err'])
    mean_errors = -np.nanmean(np.stack(pred_err, axis=0), axis=(1,2,3,4))
    
    save_path = SAVE_PATH.format(vid_id)
    output_path = OUTPUT_PATH.format(vid_id)
    output_bdy_path = OUTPUT_BDY_PATH.format(vid_id)

    if osp.exists(osp.dirname(save_path)):
        shutil.rmtree(osp.dirname(save_path))
    os.mkdir(osp.dirname(save_path))
    print(osp.dirname(save_path))

    if len(mean_errors)< 2:
        return
        
    bdy_idx_list = detect_boundary(mean_errors, offset=offset, sigma=LoG_sigma)
    bdy_idx_list_smt = detect_boundary(filters.gaussian_filter1d(mean_errors, smooth_factor), offset=offset, sigma=LoG_sigma)
    bdy_idx_save = {}
    bdy_idx_save['bdy_idx_list'] = bdy_idx_list
    bdy_idx_save['bdy_idx_list_smt'] = bdy_idx_list_smt
    pickle.dump(bdy_idx_save, open(output_bdy_path, "wb"))
    
    
if __name__ == "__main__":
    
    main(sys.argv[1])