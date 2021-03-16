## Generic Event Boundary Detection: A Benchmark for Event Segmentation 

We release our data annotation & baseline codes for detecting generic event boundaries in video.

Links: [[Arxiv](https://arxiv.org/abs/2101.10511)]  [[LOVEU Challenge](https://sites.google.com/view/loveucvpr21/home)]

Contributors: Mike Zheng Shou, Stan Lei, Deepti Ghadiyaram, Weiyao Wang, Matt Feiszli.



### Overview

This repo has the following structure:

```
./
|   LICENSE
|   README.md
|   
+---BdyDet
|   +---k400
|   |   |   detect_event_boundary.py
|   |   |   run_multiprocess_detect_event_boundary.py
|   |      
|   \---TAPOS
|       |   detect_event_boundary.py
|       |   run_multiprocess_detect_event_boundary.py
|
+---data
|   +---export
|   |   |   hmdb_mr345_min_change_duration0.3.pkl
|   |   |   hmdb_raw_annotation.pkl
|   |   |   k400_mr345_train_min_change_duration0.3.pkl
|   |   |   k400_mr345_valnew_min_change_duration0.3.pkl
|   |   |   k400_mr345_val_min_change_duration0.3.pkl
|   |   |   k400_train_raw_annotation.pkl
|   |   |   k400_valnew_raw_annotation.pkl
|   |   |   k400_val_raw_annotation.pkl
|   |   |   prepare_hmdb_release.ipynb
|   |   |   prepare_k400_release.ipynb
|   |   |   tapos_annotation_timestamps_myfps.json
|   |           
|   +---exp_k400
|   |   |   classInd.txt
|   |   |   detect_seg.eval.mindur0.3.npy
|   |   |   val_set.csv
|   |   |   window_lists.pkl
|   |   | 
|   |   +---pred_err
|   |   |   ...
|   |   |   
|   |   \---detect_seg
|   |           --07WQ2iBlw.pkl
|   |           ...
|   |           __wsytoYy3Q.pkl
|   |           
|   \---exp_TAPOS
|       |   detect_seg.eval.rgb.npy
|       |   train_set.csv
|       |   val_set.csv
|       |   window_lists.rgb.pkl
|       | 
|       +---pred_err
|       |   ...
|       |    
|       \---detect_seg
|               01d8r4klM5w_s00001_10_807_41_426.pkl
|               ...
|               CidU2e7AOOw_s00001_5_700_21_300.pkl
|               
+---eval
|   |   eval_GEBD_k400.ipynb
|   |   eval_GEBD_TAPOS.ipynb
|           
\---PA_DPC
    |   LICENSE
    |   README.md
    |   
    +---asset
    |       arch.png
    |       
    +---backbone
    |   |   convrnn.py
    |   |   resnet_2d.py
    |   |   resnet_2d3d.py
    |   |   select_backbone.py
    |           
    +---dpc
    |   |   dataset_3d_infer_pred_error.py
    |   |   main_infer_pred_error.py
    |   |   model_3d.py
    |           
    \---utils
        |   augmentation.py
        |   utils.py
```
Note that we release codes on Github. Annotations are available on [GoogleDrive](linkhere). Run the code by yourself to generate the output files.


* `data/`:
  - `export/` folder stores temporal boundary annotations of our Kinetics-GEBD and HMDB-GEBD datasets; 
  - `exp_k400/` and `exp_TAPOS` store intermediate experimental data and final results.

* `BdyDet/`: codes for detecting boundary positions based on predictability sequence.

* `eval/`: codes for evaluating the performance of boundary detection.

* `PA_DPC`: codes for computing the predictability sequence using various methods.

  


### `data/`

* In `data/export/`: 
  - `*_raw_annotation.pkl` stores the raw annotations; 
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
  -  `detect_seg/` stores output of `BdyDet/` i.e. detected boundary positions.


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

      
### Misc
* Download datasets from [Kinetics-400](https://deepmind.com/research/open-source/kinetics) and [TAPOS](https://sdolivia.github.io/TAPOS/). (Note that some of the videos can not be downloaded from YouTube for some reason, you can go ahead with those available.)

* Note that for TAPOS, you need to cut out each action instance by yourself first and then can use our following codes to process each instance's video separately.

* Extract frames of videos in the dataset.

* To reproduce PA_DPC, generate your own `data/exp_*/val_set.csv`, in which path to the folder of video frames and the number of frames in that specific video should be contained. 




### Q&A

For any questions, welcome to create an issue or email Mike (mike.zheng.shou@gmail.com) and Stan (leiwx52@gmail.com). Thank you for helping us improve our data & codes.



