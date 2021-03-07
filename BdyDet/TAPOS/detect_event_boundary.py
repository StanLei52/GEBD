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
    
    #if need to write video, you need to set the correct frame image path
    IMAGE_PATH = '....../04_26_20/frames256_instances/{}/{}/image_{:05d}.jpg'
    write_video = False
    fps = 6
    
    # deal with index/position
    offset = 4 # this is after downsample, the end frame index of the current segment (4-th) of the first window
    downsample = 3
    offset *= downsample
    
    # hyper-para for boundary detection
    smooth_factor = 5
    LoG_sigma = 15

    # Update these
    exp_path = '../../data/exp_TAPOS'
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


    # load GT boundary timestamps and convert into index here
    with open('../../data/export/tapos_annotation_timestamps_myfps.json', 'r') as f:
        tapos_dict = json.load(f)
        
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
    
    if write_video is False:
        return
        
    writer_og = VideoWriter('{img_dir}/{vid_id}_og.mp4'.format(img_dir=osp.dirname(osp.dirname(SAVE_PATH)),
                                                               vid_id=vid_id), fps=fps)

    # e.g. vid 2IO8DO_QbRE_s00018_5_324_13_160
    v = vid_id[:11]
    s = vid_id[12:]
    myfps = tapos_dict[v][s]['myfps']
    instance_start_idx = tapos_dict[v][s]['substages_myframeidx'][0]
    bdy_idx_list_gt = []
    for idx in tapos_dict[v][s]['substages_myframeidx'][1:-1]:
        bdy_idx_list_gt.append(int((idx - instance_start_idx)/downsample))
    
    for j in tqdm.tqdm(range(0,data['vlen'],downsample)):
        x = np.arange(offset, offset + len(mean_errors))
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(nrows=5, ncols=2)
        ax1 = fig.add_subplot(gs[0, 0])

        # Signal is in mean_errors.
        ax1.plot(x, filters.gaussian_filter1d(mean_errors, smooth_factor), lw=2)
        ax1.axvline(j/downsample, color='red')
        for bdy_idx in bdy_idx_list_gt: # plot GT boundaries
            ax1.axvline(bdy_idx, color='cyan', ls='-')
        for bdy_idx in bdy_idx_list_smt: # plot our detected boundaries
            ax1.axvline(bdy_idx, color='green', ls='--')
        ax1.set_xlim(0, len(mean_errors) + offset + offset_end)
        ax1.set_title('Smooth Pred Acc')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x, mean_errors, lw=2)
        ax2.axvline(j/downsample, color='red')
        for bdy_idx in bdy_idx_list_gt: # plot GT boundaries
            ax2.axvline(bdy_idx, color='cyan', ls='-')
        for bdy_idx in bdy_idx_list:
            ax2.axvline(bdy_idx, color='green', ls='--')
        ax2.set_title('Pred Acc')
        ax2.set_xlim(0, len(mean_errors) + offset + offset_end)
        
        ax4 = fig.add_subplot(gs[1, 0])
        LoG_smt_mean_errors = ndimage.gaussian_laplace(filters.gaussian_filter1d(mean_errors, smooth_factor), sigma=LoG_sigma, mode='nearest')
        ax4.plot(x, LoG_smt_mean_errors, lw=2)
        ax4.axvline(j/downsample, color='red')
        for bdy_idx in bdy_idx_list_gt: # plot GT boundaries
            ax4.axvline(bdy_idx, color='cyan', ls='-')
        for bdy_idx in bdy_idx_list_smt:
            ax4.axvline(bdy_idx, color='green', ls='--')
        ax4.set_title('LoG Smooth Pred Acc')
        ax4.set_xlim(0, len(mean_errors) + offset + offset_end)
        
        ax5 = fig.add_subplot(gs[1, 1])
        LoG_mean_errors = ndimage.gaussian_laplace(mean_errors, sigma=LoG_sigma, mode='nearest')
        ax5.plot(x, LoG_mean_errors, lw=2)
        ax5.axvline(j/downsample, color='red')
        for bdy_idx in bdy_idx_list_gt: # plot GT boundaries
            ax5.axvline(bdy_idx, color='cyan', ls='-')
        for bdy_idx in bdy_idx_list:
            ax5.axvline(bdy_idx, color='green', ls='--')
        ax5.set_title('LoG Pred Acc')
        ax5.set_xlim(0, len(mean_errors) + offset + offset_end)
        
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(x, np.gradient(LoG_smt_mean_errors), lw=2)
        ax6.axvline(j/downsample, color='red')
        for bdy_idx in bdy_idx_list_gt: # plot GT boundaries
            ax6.axvline(bdy_idx, color='cyan', ls='-')
        for bdy_idx in bdy_idx_list_smt:
            ax6.axvline(bdy_idx, color='green', ls='--')
        ax6.axhline(0, color='green', ls='--')
        ax6.set_title('$\Delta$ LoG Smooth Pred Acc')
        ax6.set_xlim(0, len(mean_errors) + offset + offset_end)
        
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(x, np.gradient(LoG_mean_errors), lw=2)
        ax7.axvline(j/downsample, color='red')
        for bdy_idx in bdy_idx_list_gt: # plot GT boundaries
            ax7.axvline(bdy_idx, color='cyan', ls='-')
        for bdy_idx in bdy_idx_list:
            ax7.axvline(bdy_idx, color='green', ls='--')
        ax7.axhline(0, color='green', ls='--')
        ax7.set_title('$\Delta$ LoG Pred Acc')
        ax7.set_xlim(0, len(mean_errors) + offset + offset_end)

        ax3 = fig.add_subplot(gs[-2:, :])
        im = plt.imread(IMAGE_PATH.format(v, s, (j + 1)))
        cv2.putText(im, str(j), (20, 20), 0, 1, (57, 255, 20), thickness=2)
        writer_og.add_image(im)
        ax3.imshow(im)
        ax3.axis('off')
        

        plt.tight_layout()
        plt.savefig(
            fname=save_path.format(int(j/downsample)),
            dpi=150
        )
        plt.close()
    cmd = [
        'ffmpeg',
        '-y',
        '-threads', '16',
        '-framerate', str(fps),
        '-i', '{img_dir}/frame_%05d.jpg'.format(img_dir=osp.dirname(save_path)),
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-an',
        # Note that if called as a string, ffmpeg needs quotes around the
        # scale invocation.
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        output_path,
    ]
    print(' '.join(cmd))
    try:
        err = subprocess.call(cmd)
        if err:
            print('Subprocess failed')
    except OSError:
        print('Failed to run ffmpeg')
    writer_og.make_video()
    writer_og.close()

    
    
if __name__ == "__main__":
    
    main(sys.argv[1])