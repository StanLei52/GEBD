# PC - GEBD

Here, we release codes for PC -- an end-to-end training method for GEBD. 

In the PC framework, for each frame _f_ in a video, we take _T_ frames preceeding _f_ and _T_ frames suceeding _f_ as inputs, and then build a binary classifier  to predict if _f_ is boundary or background.

Refer to our [paper](https://arxiv.org/abs/2101.10511) for more details.



## Get Started

- Check `datasets/MultiFDataset.py` to generate GT files for training. Note that you should prepare `k400_mr345_*SPLIT*_min_change_duration0.3.pkl` for Kinetics-GEBD and `TAPOS_*SPLIT*_anno.pkl`  (this should be organized as `k400_mr345_*SPLIT*_min_change_duration0.3.pkl` for convenience) for TAPOS before running our code. 

-  Accordingly change `PATH_TO` in our codes to your data/frames path as needed.

- Train on Kinetics-GEBD:

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

- Train on TAPOS

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



## New Features

- Distributed Balance Batch Sampler, see `utils/sampler.py`.

  

## Models

- Will be released soon.

  

## Acknowledgement 

- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman).
- Thank [Mike Shou](http://www.columbia.edu/~zs2262/) for his insights.



## Q&A

For any questions, welcome to create an issue or email Stan(leiwx52@gmail.com).
