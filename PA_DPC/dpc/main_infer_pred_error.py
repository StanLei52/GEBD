import os
import ipdb
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle

sys.path.append('../utils')
from dataset_3d_infer_pred_error import *
from model_3d import *
from resnet_2d3d import neq_load_customized
import torchvision
from augmentation import *
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--pred_task', default='', type=str)
parser.add_argument('--pkl_folder_name', default='', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--unit_test', default=0, type=int)
parser.add_argument('--mode', default='all', type=str)
parser.add_argument('--modality', default='rgb', type=str)
parser.add_argument('--which_split', default=1, type=int)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')
    
    ### output file directory ###
    global pkl_folder
    if args.pkl_folder_name != '': # allow user to explicitly specify output dir; otherwise follow default in the below 
        pkl_folder = args.pkl_folder_name + '/pred_err'
    else:
        if args.dataset == 'tapos_instances':
            pkl_folder = '../../data/exp_TAPOS/' + args.pred_task + '/pred_err'
        elif args.dataset == 'k400':
            pkl_folder = '../../data/exp_k400/' + args.pred_task + '/pred_err'
        args.pkl_folder_name = pkl_folder[:-9]

    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)
    

    ### dpc model ###
    if args.model == 'dpc-rnn':
        model = DPC_RNN_Infer_Pred_Error(sample_size=args.img_dim, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len, 
                        network=args.net, 
                        pred_step=args.pred_step)
    elif args.model == 'resnet50-beforepool':
        model = ResNet50_BeforePool_Feature_Extractor(pretrained=True, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len,  
                        pred_step=args.pred_step)
    elif args.model == 'resnet18-beforepool':
        model = ResNet18_BeforePool_Feature_Extractor(pretrained=True, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len,  
                        pred_step=args.pred_step)
    elif args.model == 'rgb-avg-temporal':
        model = RBG_Extractor(pretrained=True, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len,  
                        pred_step=args.pred_step)
    else: raise ValueError('wrong model!')

    model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion; criterion = nn.MSELoss(reduction='none')

    print('\n===========No grad for all layers============')
    for name, param in model.named_parameters():
        param.requires_grad = False
        print(name, param.requires_grad)
    print('==============================================\n')

    params = model.parameters()

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else: 
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    transform = transforms.Compose([
        Scale(size=(args.img_dim,args.img_dim)),
        ToTensor(),
        Normalize()
    ])
        
    val_loader = get_data(transform, args.mode)
    
    ### main loop ###
    validate(val_loader, model)


def validate(data_loader, model):
    model.eval()

    with torch.no_grad():
        for idx, (input_seq, video_id, vlen, 
                  window_idx, current_frame_idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            #ipdb.set_trace()
            [target_, pred_] = model(input_seq) # each output of shape [B, pred_step, D, last_size, last_size]
            del input_seq
            loss = criterion(pred_, target_).cpu() # default mean
            
            # save prediction error seq
            for idx in range(len(video_id)):
                err_dict = {}
                err_dict['pred_err'] = loss[idx].numpy()
                err_dict['video_id'] = video_id[idx]
                err_dict['vlen'] = vlen[idx].cpu().numpy()
                err_dict['window_idx'] = int(window_idx[idx].cpu().numpy())
                err_dict['current_frame_idx'] = int(current_frame_idx[idx].cpu().numpy())
                    
                pkl_name = pkl_folder + '/err_vid_' + video_id[idx] + '_currentframe_' + str(int(err_dict['current_frame_idx'])).zfill(6) + '.pkl'
                #ipdb.set_trace()
                pickle.dump(err_dict, open(pkl_name, "wb"))


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    use_big = args.img_dim > 140
    if args.unit_test == 0:
        unit_test = False
    else:
        unit_test = args.unit_test
    if args.dataset == 'k400':
        dataset = Kinetics400_full_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=args.ds,
                              big=use_big,
                              unit_test=unit_test,
                              modality=args.modality,
                              pkl_folder_name=args.pkl_folder_name,
                              pred_task=args.pred_task,
                              pred_step=args.pred_step)
    elif args.dataset == 'tapos_instances':
        dataset = TAPOS_instances_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=args.ds,
                              big=use_big,
                              unit_test=unit_test,
                              modality=args.modality,
                              pkl_folder_name=args.pkl_folder_name,
                              pred_task=args.pred_task,
                              pred_step=args.pred_step)
    else:
        raise ValueError('dataset not supported')

    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=None,
                                  shuffle=False,
                                  num_workers=32,
                                  pin_memory=True)
    
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


if __name__ == '__main__':
    main()
