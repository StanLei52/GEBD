import os
import argparse
from os.path import abspath
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging
from datetime import datetime
from collections import OrderedDict

from utils.getter import getModel, getDataset, getDataLoader_for_test, data_prefetcher
from utils.helper import resume_checkpoint, load_checkpoint, get_outdir
from utils.metric import AverageMeter, accuracy

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Training Config', add_help=False)
# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='kinetics_multiframes',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='val',
                    help='dataset validation split (default: validation)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--model', default='multiframes_resnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                    help='number of label classes (Model default if None)')


parser.add_argument('--pred-output', default='./multif-pred_outputs', type=str, help='outp for predtion')
args = parser.parse_args()

def main():
    print(args)
    test_dataset = getDataset(args.dataset, mode=args.val_split, args=args)

    dataloader = getDataLoader_for_test(test_dataset, args=args)
    model = getModel(model_name=args.model, args=args)

    print('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    # move model to gpu
    model.cuda()

    assert args.resume is not None, ('Resume path should not be none for evaluating models.')
    resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None,
            loss_scaler=None,
            log_info=False
        )
    print(f'Model loaded from epoch {resume_epoch}.')
    
    model_pred_dict = dict()
    model.eval()
    with torch.no_grad():
        for i,items in tqdm(enumerate(dataloader), total=len(dataloader)):
            inps, paths = items['inp'], items['path']
            outps = model(inps.cuda(non_blocking=True))
            if isinstance(outps,(list, tuple)):
                outps = outps[0]

            bdy_scores = F.softmax(outps, dim=1)[:,1].cpu().numpy()
            writing_scores(paths, bdy_scores, model_pred_dict, args)
    
    if not os.path.exists(args.pred_output):
        os.makedirs(args.pred_output)
    
    filename = '{}_{}_{}_{}_scores.pkl'.format(args.resume.replace('/','_'), args.val_split, resume_epoch, args.dataset)
    print(filename)
   
    with open(os.path.join(args.pred_output, filename),'wb') as f:
        pickle.dump(model_pred_dict,f,pickle.HIGHEST_PROTOCOL)
    
    del model_pred_dict



def writing_scores(abs_path, scores, dict_record, args):
    for path,score in zip(abs_path, scores): 
        #/Checkpoint/leiwx/weixian/data/Kinetics_GEBD_frame/val_split/waxing_chest/Swo0mvCOPz4_000235_000245/image_00231.jpg' 
        #/Checkpoint/leiwx/weixian/TAPOS/instances_frame256/oxjUeTc_Vag/s00019_2_603_17_620/image_00231.jpg
        frame = path.split('/')[-1]
        frame_idx = int(frame[6:11])
        vid = get_vid_from_path(args=args, path=path)
        if vid not in dict_record.keys():
            dict_record[vid]=dict()
            dict_record[vid]['frame_idx']=[]
            dict_record[vid]['scores']=[]
        dict_record[vid]['frame_idx'].append(frame_idx)
        dict_record[vid]['scores'].append(score)

def get_vid_from_path(args,path):
    if 'kinetics' in args.dataset.lower():
        vid_dir, _ = path.split('/')[-2:]
        vid = vid_dir[:11]
        return vid
    elif 'tapos' in args.dataset.lower():
        vid = '_'.join(path.split('/')[-3:-1])
        return vid
    else:
        raise NotImplementedError



if __name__ == '__main__':
    main()
