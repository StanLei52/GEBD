## Instructions for LOVEU@CVPR'2021 Challenge Participants

Here, we give some instructions on how to prepare data and submission files for [LOVEU@CVPR'2021 Challenge](https://sites.google.com/view/loveucvpr21/home). The examples in this material are based on the Kinetics-GEBD dataset.

### 1. Data Preparation

**1-a**. Download [Kinetics-GEBD annotation](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo): `k400_train_raw_annotation.pkl` and `k400_val_raw_annotation.pkl`.

**1-b**. Download videos listed in the [Kinetics-GEBD annotation](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo). Note that videos in the Kinetics-GEBD dataset are a subset of Kinetics-400 dataset. You can either download the whole Kinetics-400 dataset or just download the part in Kinetics-GEBD dataset.

**1-c**. Trim the videos according to the Kinetics-400 annotations. E.g., after you downloading `3y1V7BNNBds.mp4`, trim this video into a 10-second video `3y1V7BNNBds_000000_000010.mp4` from 0s to 10s in the original video. Note that the start time and end time for each video can be found at the Kinetics-400 annotations.

**1-d**. Extract frames. You can follow the instructions [DPC/process_data](https://github.com/TengdaHan/DPC/tree/master/process_data) to extract frames of the trimmed videos.

**1-e**. Generate GT files `k400_mr345_train_min_change_duration0.3.pkl` and `k400_mr345_val_min_change_duration0.3.pkl` . Refer to  [prepare_k400_release.ipynb](https://github.com/StanLei52/GEBD/blob/main/data/export/prepare_k400_release.ipynb) for generating the GT files. Specifically, you should prepare the <u>train</u> and <u>val</u> split:

```python
generate_frameidx_from_raw(split='train')
generate_frameidx_from_raw(split='val')
```



### 2. Train your model

In this part, you can customize your own training procedure. We take our `PC` baseline as example here:

**2-a**. Check `PC/datasets/MultiFDataset.py` to generate files for training.

**2-b**. Accordingly change `PATH_TO` in our codes to your data/frames path as needed. Note that your data should be prepared as stated in **step 1**.

**2-c**. Train on Kinetics-GEBD:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 PC_train.py \
--dataset kinetics_multiframes \
--train-split train \
--val-split valnew \
--num-classes 2 \
--batch-size 32 \
--n-sample-classes 2 \
--n-samples 16 \
--lr 0.01 \
--warmup-epochs 0 \
--epochs 30 \
--decay-epochs 10 \
--model multiframes_resnet \
--pin-memory \
--balance-batch \
--sync-bn \
--amp \
--native-amp \
--eval-metric loss \
--log-interval 50 
```

**2-d**. Generate the sequence score for videos in validation set with checkpoint:

```shell
CUDA_VISIBLE_DEVICES=0 python PC_test.py \ 
--dataset kinetics_multiframes \
--val-split val \
--resume path_to/checkpoint
```



### 3. Generate Submission File.

Participants are required to submit their results in a pickle file. The pickle format is composed of a dictionary containing keys with video identifiers and values with boundary lists. For example,  

```shell
{
‘6Tz5xfnFl4c’: [5.9, 9.4], # boundaries detected at 5.9s, 9.4s of this video
‘zJki61RMxcg’: [0.6, 1.5, 2.7] # boundaries detected at 0.6s, 1.5s, 2.7s of this video
...
}
```

Generally, you should generate such detect boundaries according to your own configuration. All you need is to generate the detected boundaries for each video. For our `PC` baseline, we obtained a score sequence for each video after executing `PC_test.py`. For example,

```shell
{
'S4hYqZbu0kQ': {'frame_idx': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99], 'scores': [0.87680495, 0.89294374, 0.7491671, 0.685366, 0.3733009, 0.118334584, 0.1789392, 0.32489958, 0.39417332, 0.59364367, 0.78907657, 0.38130152, 0.14496064, 0.07595703, 0.069147676, 0.017675582, 0.048309233, 0.15596619, 0.17298242, 0.046014242, 0.12774284, 0.20063624, 0.20420128, 0.41636664, 0.7791878, 0.8065185, 0.80875623, 0.794678, 0.6545196, 0.54669106, 0.60019726, 0.6882232, 0.52349806]}
...
}
```

For this video, we recorded the frame index after downsampling (`ds=3`) and the corresponding score. One can determine the position of boundaries according to the score sequence. A simple way is to set a threshold(i.e. 0.5) to filter out some low probability timestamps and keep those high. For our baseline, we group consecutive frames with high probability(i.e. 0.5) and mark their center as a boundary; also you can try to use Gaussian smoothing, etc.

We provide an example to obtain the submission file (just for reference).

**3-a**. Generate submission file from sequence score:

```python
import pickle
import numpy as np

def get_idx_from_score_by_threshold(threshold=0.5, seq_indices=None, seq_scores=None):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    for i in range(len(seq_scores)):
        if seq_scores[i]>=threshold:
            internals_indices.append(i)
        elif seq_scores[i]<threshold and len(internals_indices)!=0:
            bdy_indices.append(internals_indices)
            internals_indices=[]
            
        if i==len(seq_scores)-1 and len(internals_indices)!=0:
            bdy_indices.append(internals_indices)
            
    bdy_indices_in_video = []
    if len(bdy_indices)!=0: 
        for internals in bdy_indices:
            center = int(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
    return bdy_indices_in_video

with open('/path_to/k400_mr345_val_min_change_duration0.3_with_timestamps.pkl', 'rb') as f:
    gt_dict = pickle.load(f, encoding='lartin1')
    
with open('/path_to/score_sequence.pkl','rb') as f: #
    my_pred = pickle.load(f,encoding='lartin1')

print(len(gt_dict))
save = dict()
for vid in my_pred:
    if vid in gt_dict:
      	# detect boundaries, convert frame_idx to timestamps
        fps = gt_dict[vid]['fps']
        det_t = np.array(get_idx_from_score_by_threshold(threshold=0.5, 
                                                        seq_indices=my_pred[vid]['frame_idx'], 
                                                        seq_scores=my_pred[vid]['scores']))/fps
        save[vid] = det_t.tolist()
print(len(save))
pickle.dump(save,open('submission.pkl','wb'),protocol=pickle.HIGHEST_PROTOCOL)
```




