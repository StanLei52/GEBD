## Generic Event Boundary Detection: A Benchmark for Event Segmentation 

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

We release our data annotation & baseline codes for detecting generic event boundaries in video.

Links: [[Arxiv](https://arxiv.org/abs/2101.10511)]  [[LOVEU Challenge](https://sites.google.com/view/loveucvpr21/home)]

Contributors: Mike Zheng Shou, Stan Lei, Deepti Ghadiyaram, Weiyao Wang, Matt Feiszli.



### Overview

This repo has the following structure:

```
./
│   LICENSE
│   README.md
│   INSTRUCTIONS.md
│
├───BdyDet
│   ├───k400
│   │       detect_event_boundary.py
│   │       run_multiprocess_detect_event_boundary.py
│   │
│   └───TAPOS
│           detect_event_boundary.py
│           run_multiprocess_detect_event_boundary.py
│
├───Challenge_eval_Code
│       eval.py
│       README.md
│
├───data
│   ├───export
│   │       prepare_hmdb_release.ipynb
│   │       prepare_k400_release.ipynb
│   │
│   ├───exp_k400
│   │   │   classInd.txt
│   │   │   val_set.csv
│   │   │
│   │   ├───detect_seg
│   │   └───pred_err
│   └───exp_TAPOS
│       │   train_set.csv
│       │   val_set.csv
│       │
│       ├───detect_seg
│       └───pred_err
├───eval
│       eval_GEBD_k400.ipynb
│       eval_GEBD_TAPOS.ipynb
│
├───PA_DPC
│   │   LICENSE
│   │   README.md
│   │
│   ├───asset
│   │       arch.png
│   │
│   ├───backbone
│   │       convrnn.py
│   │       resnet_2d.py
│   │       resnet_2d3d.py
│   │       select_backbone.py
│   │
│   ├───dpc
│   │       dataset_3d_infer_pred_error.py
│   │       main_infer_pred_error.py
│   │       model_3d.py
│   │
│   └───utils
│           augmentation.py
│           utils.py
│
└───PC
    │   PC_test.py
    │   PC_train.py
    │   README.md
    │
    ├───DataAssets
    ├───datasets
    │       augmentation.py
    │       MultiFDataset.py
    │
    ├───modeling
    │       resnetGEBD.py
    │
    ├───run
    │       pc_k400_dist.sh
    │       pc_tapos_dist.sh
    │
    └───utils
            augmentation.py
            checkpoint_saver.py
            augmentation.py
            checkpoint_saver.py
            clip_grad.py
            cuda.py
            getter.py
            helper.py
            log.py
            metric.py
            model_ema.py
            optim_factory.py
            sampler.py
            scheduler.py
```

Note that we release codes on Github. Annotations are available on [GoogleDrive](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo?usp=sharing). Run the code by yourself to generate the output files. **Refer to [INSTRUCTIONS](INSTRUCTIONS.md) for preparing data and generating submission files.**


* `data/`:

  - `export/` folder stores temporal boundary annotations of our Kinetics-GEBD and HMDB-GEBD datasets; download our raw annotations and put them under this folder.
  - `exp_k400/` and `exp_TAPOS/` store intermediate experimental data and final results.

* `Challenge_eval_Code/`: codes for evaluation in LOVEU Challenge Track 1.

* `BdyDet/`: codes for detecting boundary positions based on predictability sequence.

* `eval/`: codes for evaluating the performance of boundary detection.

* `PA_DPC/`: codes for computing the predictability sequence using various methods.

* `PC/`: codes for supervised baseline on GEBD.

  


### `data/`

* In `data/export/`: 
  - `*_raw_annotation.pkl` stores the raw annotations; download raw annotations [here](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo?usp=sharing).
  - we further filter out videos that receives <3 annotations and conduct pre-processing e.g. merge very close boundaries - we use notebook `prepare_*_release.ipynb` and the output is stored in `*_mr345_min_change_duration0.3.pkl`. 

* Some fields in `*_raw_annotation.pkl`:
  - `fps`: frames per second.
  - `video_duration`:video duration in second.
  - `f1_consis`: a list of consistency scores, each score corresponds to a specific annotator’s score as compared to other annotators.
  - `substages_timestamps`: a list of annotations from each annotator; each annotator’s annotation is again a list of boundaries. time in second; for 'label', it is of format `A: B`.
    + If A is “ShotChangeGradualRange”, B could be “Change due to Cut”, “Change from/to slow motion”, “Change from/to fast motion”, or “N/A”.
    + If A is “ShotChangeImmediateTimestamp”,  B could be “Change due to Pan”, “Change due to Zoom”, “Change due to Fade/Dissolve/Gradual”, “Multiple”, or “N/A”.
    + If A is “EventChange (Timestamp/Range)”, B could be “Change of Color”, “Change of Actor/Subject”, “Change of Object Being Interacted”, “Change of Action”, “Multiple”, or “N/A”.

* Some fields in `*_mr345_min_change_duration0.3.pkl`:
  - `substages_myframeidx`: the term of substage is following the convention in TAPOS to refer to boundary position -- here we store each boundary’s frame index which starts at 0.


* In `data/exp_k400/` and `data/exp_TAPOS/`: 
  - `pred_err/` stores output of `PA_DPC/` i.e. the predictability sequence;
  - `detect_seg/` stores output of `BdyDet/` i.e. detected boundary positions.



### `Challenge_eval_Code/`

- We use `eval.py` for evaluation in our competition.

- Although one can use frame_index and number of total frames to measure the *Rel.Dis*, as we implemented in `eval/`, you should **represent the detected boundaries with timestamps (in seconds)**. For example:

  ```shell
  {
  ‘6Tz5xfnFl4c’: [5.9, 9.4], # boundaries detected at 5.9s, 9.4s of this video
  ‘zJki61RMxcg’: [0.6, 1.5, 2.7] # boundaries detected at 0.6s, 1.5s, 2.7s of this video
  ...
  }
  ```

- Refer to [this file](https://github.com/StanLei52/GEBD/blob/main/data/export/prepare_k400_release.ipynb) to generate GT files from raw annotations.




### `BdyDet/`

Change to directory  `./BdyDet/k400` or `./BdyDet/TAPOS` and run the following command, which will launch multiple processes of `detect_event_boundary.py`.

(Note to set the number of processes according to your server in `detect_event_boundary.py`.)

  ```shell
python run_multiprocess_detect_event_boundary.py
  ```



### `PA_DPC/`

Our implementation is based on the [[DPC](https://github.com/TengdaHan/DPC)] framework. Please refer to their README or website to learn installation and usage. In the below, we only explain how to run our scripts and what are our modifications.

* Modifications at a glance

  - `main_infer_pred_error.py` runs in only inference mode to assess predictability over time.
  - `dataset_3d_infer_pred_error.py` contains loaders for two datasets i.e. videos from Kinetics and truncated instances from TAPOS.
  - `model_3d.py` adds several classes for feature extraction purposes, e.g. ResNet feature before the pooling layer, to enable computing feature difference directly based on ImageNet pretrained model, in contrast to the predictive model in DPC.

* How to run?

  change to directory  `./PA_DPC/dpc/`

  - Kinetics-GEBD:

  ```shell
  python main_infer_pred_error.py --gpu 0 --model resnet50-beforepool --num_seq 2 --pred_step 1 --seq_len 5 --dataset k400 --batch_size 160 --img_dim 224 --mode val --ds 3 --pred_task featdiff_rgb
  ```

  - TAPOS:

  ```shell
  python main_infer_pred_error.py --gpu 0 --model resnet50-beforepool --num_seq 2 --pred_step 1 --seq_len 5 --dataset tapos_instances --batch_size 160 --img_dim 224 --mode val --ds 3 --pred_task featdiff_rgb
  ```

  - Notes:

    - `ds=3 ` means we downsample the video frames by 3 to reduce computation.

    - `seq_len=5` means 5 frames for each _step_ (concept used in DPC).

    - `num_seq=2`and ` pred_step=1` so that at a certain time position _t_, we use 1 step before and 1 step after to assess the predictability at time _t_; thus the predictability at time _t_ is the feature difference between the average feature of 5 sampled frames before _t_ and the average feature of 5 sampled frames after _t_.

    - we slide such model over time to obtain predictability at different temporal positions. More details can be found in `dataset_3d_infer_pred_error.py`. Note that `window_lists.pkl` stores the index of frames to be sampled for every window - since it takes a long time to compute (should be able to optimize in the future), we store its value and just load this pre-computed value in the future runs on the same dataset during experimental explorations.



### `PC/`

PC is a supervised baseline for GEBD task. In the PC framework, for each frame _f_ in a video, we take _T_ frames preceding _f_ and _T_ frames succeeding _f_ as inputs, and then build a binary classifier  to predict if _f_ is boundary or background.

- Get Started

  - Check `PC/datasets/MultiFDataset.py` to generate GT files for training. Note that you should prepare `k400_mr345_*SPLIT*_min_change_duration0.3.pkl` for Kinetics-GEBD and `TAPOS_*SPLIT*_anno.pkl`  (this should be organized as `k400_mr345_*SPLIT*_min_change_duration0.3.pkl` for convenience) for TAPOS before running our code.
  - You should accordingly change `PATH_TO` in our codes to your data/frames path as needed.

- How to run?

  Change directory to `./PC`.

  - Train on Kinetics-GEBD:

    ```shell
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 PC_train.py \
    --dataset kinetics_multiframes \
    --train-split train \
    --val-split val \
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

  - Train on TAPOS:

    ```shell
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 PC_train.py \
    --dataset tapos_multiframes \
    --train-split train \
    --val-split val \
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

  - Generate scores sequence on Kinetics-GEBD Validation Set:

    ```shell
    CUDA_VISIBLE_DEVICES=0 python PC_test.py \ 
    --dataset kinetics_multiframes \
    --val-split val \
    --resume path_to/checkpoint
    ```

  - Generate scores sequence on TAPOS:

    ```shell
    CUDA_VISIBLE_DEVICES=0 python PC_test.py \ 
    --dataset tapos_multiframes \
    --val-split val \
    --resume path_to/checkpoint
    ```

- Models

  - [k400](https://drive.google.com/file/d/1jtFag6EmkPpdlsvgusTjRpRW3xWTDBuI/view?usp=sharing)



### Misc

* Download datasets from [Kinetics-400](https://deepmind.com/research/open-source/kinetics) and [TAPOS](https://sdolivia.github.io/TAPOS/). (Note that some of the videos can not be downloaded from YouTube for some reason, you can go ahead with those available.)

* Note that for TAPOS, you need to cut out each action instance by yourself first and then can use our following codes to process each instance's video separately.

* Extract frames of videos in the dataset.

* To reproduce PA_DPC, generate your own `data/exp_*/val_set.csv`, in which path to the folder of video frames and the number of frames in that specific video should be contained. 

### Cite our work
```
@InProceedings{Shou_GEBD_2021_ICCV,
    author    = {Shou, Mike Zheng and Lei, Stan Weixian and Wang, Weiyao and Ghadiyaram, Deepti and Feiszli, Matt},
    title     = {Generic Event Boundary Detection: A Benchmark for Event Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8075-8084}
}
```


### Q&A

For any questions, welcome to create an issue or email Mike (mike.zheng.shou@gmail.com) and Stan (leiwx52@gmail.com). Thank you for helping us improve our data & codes.
