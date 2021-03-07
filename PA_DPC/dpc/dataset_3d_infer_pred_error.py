import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import ipdb
import pandas as pd
import numpy as np
import cv2
sys.path.append('../utils')
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

        
def pil_loader_flow(path_x, path_y):
    with open(path_x, 'rb') as f:
        with Image.open(f) as img:
            x = np.asarray(img)
            x = np.round((x-127.5)*127.5/20+127.5)
            x = (np.minimum(np.maximum(x, 0.0), 255.0))
    with open(path_y, 'rb') as f:
        with Image.open(f) as img:
            y = np.asarray(img)
            y = np.round((y-127.5)*127.5/20+127.5)
            y = (np.minimum(np.maximum(y, 0.0), 255.0))
    img = 128*np.ones((x.shape[0], x.shape[1],3), dtype=np.uint8)
    img[:, :, 0] = x
    img[:, :, 1] = y
    return Image.fromarray(img).convert('RGB')

    
    
class TAPOS_instances_3d(data.Dataset):
    def __init__(self,
                 mode='val',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 modality='rgb',
                 pkl_folder_name='',
                 big=False,
                 return_label=False,
                 pred_task='',
                 pred_step=3):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.unit_test = unit_test
        self.return_label = return_label
        self.pred_step = pred_step
        self.pred_task = pred_task
        self.modality = modality
        self.pkl_folder_name = pkl_folder_name

        if big: print('Using TAPOS (224x224)')
        else: print('Using TAPOS (112x112) ')

        # splits
        if big:
            if mode == 'train':
                raise ValueError('train mode NOT IMPLEMENTED')
            elif (mode == 'val') or (mode == 'test'):
                split = '../../data/exp_TAPOS/val_set.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')
        else: # small
            raise ValueError('NOT IMPLEMENTED')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 10:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)
        print('#videos kept: ' + str(len(self.video_info)))

        if mode == 'val': self.video_info = self.video_info.sample(frac=1, random_state=666)# validate on 30% data only
        if self.unit_test: self.video_info = self.video_info.sample(32, random_state=666) # sample a few videos for unittest
        # shuffle not necessary because use RandomSampler
        
        print('construct windows of stride 1 from each video ...')
        self.window_info = pd.DataFrame(columns=['vpath', 'vlen', 'window_idx',
                                                 'all_window_seq_idx_block',
                                                 'current_frame_idx'])
        
        pkl_dir = self.pkl_folder_name
        if not os.path.exists(pkl_dir): os.makedirs(pkl_dir)
        if self.mode == 'train': 
            raise ValueError('NOT IMPLEMENTED')
        else:
            pkl_name = pkl_dir+"/window_lists.rbg.pkl"
            if self.modality == 'flow':
                pkl_name = pkl_dir+"/window_lists.flow.pkl"
        if os.path.exists(pkl_name):
            print("skip constructing windows... use pre-computed one...")
            self.window_info = pickle.load(open(pkl_name, "rb"))
        else:
            # stride per sampled frame
            for _, (vpath, vlen) in tqdm(self.video_info.iterrows(), total=len(self.video_info)):
                if vlen-self.num_seq*self.seq_len*self.downsample <= 0: 
                    print("vlen: "+str(vlen))
                    print("self.num_seq*self.seq_len*self.downsample: "+str(self.num_seq*self.seq_len*self.downsample))
                    continue
                n_window = int(vlen/self.downsample) - (1+1)*self.seq_len + 1
                all_window_seq_idx_block = np.zeros((n_window, self.num_seq, self.seq_len))
                for window_idx in range(n_window):
                    start_zeropad = self.num_seq - 1 - self.pred_step
                    window_start_frame_idx = window_idx*self.downsample - start_zeropad*self.seq_len*self.downsample
                    seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + window_start_frame_idx
                    tmp = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
                    #ipdb.set_trace()
                    tmp[tmp<0] = 0
                    tmp[tmp>vlen-1] = vlen-1
                    all_window_seq_idx_block[window_idx] = tmp
                    self.window_info.loc[len(self.window_info)] = [vpath, vlen, window_idx,
                                                        all_window_seq_idx_block[window_idx],
                                                        all_window_seq_idx_block[window_idx][self.num_seq-self.pred_step-1][0]]
            print("len(self.window_info): "+str(len(self.window_info)))

            pickle.dump(self.window_info, open(pkl_name, "wb"))
              
    def __getitem__(self, index):
        vpath, vlen, window_idx, idx_block, current_frame_idx = self.window_info.iloc[index]
        if idx_block is None: print(vpath) 
        
        n_window = vlen-self.num_seq*self.seq_len*self.downsample
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        if self.modality == 'flow':
            seq = [pil_loader_flow(os.path.join(vpath, 'flow_x_%05d.jpg' % (i+1)),os.path.join(vpath, 'flow_y_%05d.jpg' % (i+1))) for i in idx_block]
        
        if self.modality == 'rgb':
            seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        
        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        videoid = vpath.split('/')[-2] + '_' + vpath.split('/')[-1]
        return t_seq, videoid, vlen, window_idx, current_frame_idx

    def __len__(self):
        return len(self.window_info)
    
    
    
class Kinetics400_full_3d(data.Dataset):
    def __init__(self,
                 mode='val',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 big=False,
                 return_label=False,
                 modality='rgb',
                 pkl_folder_name='',
                 pred_task='',
                 pred_step=3):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.unit_test = unit_test
        self.return_label = return_label
        self.pred_step = pred_step
        self.modality = modality
        self.pkl_folder_name = pkl_folder_name
        self.pred_task = pred_task

        if big: print('Using Kinetics400 GEBD data (224x224)')
        else: print('Using Kinetics400 GEBD data (112x112)')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../../data/exp_k400/', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=',', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # splits
        if big:
            if mode == 'train':
                raise ValueError('train mode NOT IMPLEMENTED')
            elif (mode == 'val') or (mode == 'test'):
                split = '../../data/exp_k400/val_set.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')
        else: 
            raise ValueError('NOT IMPLEMENTED')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 10:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)
        print('#videos kept: ' + str(len(self.video_info)))

        if mode == 'val': self.video_info = self.video_info.sample(frac=1, random_state=666)#
        if self.unit_test: self.video_info = self.video_info.sample(32, random_state=666) # sample a few videos for unittest
        # shuffle not necessary because use RandomSampler
        
        print('construct windows of stride 1 from each video ...')
        self.window_info = pd.DataFrame(columns=['vpath', 'vlen', 'window_idx',
                                                 'all_window_seq_idx_block',
                                                 'current_frame_idx'])
        
        pkl_dir = self.pkl_folder_name
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
        if self.mode == 'train': 
            raise ValueError('NOT IMPLEMENTED')
        else:
            pkl_name = pkl_dir+"/window_lists.pkl"
        if os.path.exists(pkl_name):
            print("skip constructing windows... use pre-computed one...")
            self.window_info = pickle.load(open(pkl_name, "rb"))
        else:
            # stride per sampled frame
            ct_window_info_slice = 0
            v_idx_startfrom1 = 0 #counter of showing progess
            for _, (vpath, vlen) in tqdm(self.video_info.iterrows(), total=len(self.video_info)):
                v_idx_startfrom1 += 1
                if v_idx_startfrom1==1000:
                    v_idx_startfrom1 = 0
                    pickle.dump(self.window_info, open(pkl_name+str(ct_window_info_slice), "wb"))
                    print("+1 to ct_window_info_slice: "+str(ct_window_info_slice))
                    ct_window_info_slice += 1
                    self.window_info = pd.DataFrame(columns=['vpath', 'vlen', 'window_idx',
                                                 'all_window_seq_idx_block',
                                                 'current_frame_idx'])
                if vlen-self.num_seq*self.seq_len*self.downsample <= 0: 
                    print("vlen: "+str(vlen))
                    print("self.num_seq*self.seq_len*self.downsample: "+str(self.num_seq*self.seq_len*self.downsample))
                    continue
                n_window = int(vlen/self.downsample) - (1+1)*self.seq_len + 1
                all_window_seq_idx_block = np.zeros((n_window, self.num_seq, self.seq_len))
                for window_idx in range(n_window):
                    start_zeropad = self.num_seq - 1 - self.pred_step
                    window_start_frame_idx = window_idx*self.downsample - start_zeropad*self.seq_len*self.downsample
                    seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + window_start_frame_idx
                    tmp = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
                    tmp[tmp<0] = 0
                    tmp[tmp>vlen-1] = vlen-1
                    all_window_seq_idx_block[window_idx] = tmp
                    self.window_info.loc[len(self.window_info)] = [vpath, vlen, window_idx,
                                                        all_window_seq_idx_block[window_idx],
                                                        all_window_seq_idx_block[window_idx][self.num_seq-self.pred_step-1][0]]
            # handling memory constraint
            # merge the splits generated before
            merge = []
            for i in range(ct_window_info_slice):
                with open(pkl_name+str(i), "rb") as f:
                    w = pickle.load(f, encoding='lartin1')
                    merge.append(w)
                os.remove(pkl_name+str(i))
            merge.append(self.window_info)
            self.window_info = pd.concat(merge, ignore_index=True)
            print("len(self.window_info): "+str(len(self.window_info)))
            pickle.dump(self.window_info, open(pkl_name, "wb"))
              
    def __getitem__(self, index):
        vpath, vlen, window_idx, idx_block, current_frame_idx = self.window_info.iloc[index]
        if idx_block is None: print(vpath) 
        
        n_window = vlen-self.num_seq*self.seq_len*self.downsample
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        # FIXME
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        videoid = vpath.split('/')[-1][:11]
        return t_seq, videoid, vlen, window_idx, current_frame_idx

    def __len__(self):
        return len(self.window_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]

