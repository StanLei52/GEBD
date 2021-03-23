#  Evaluation code for [LOVEU@CVPR'21 Challenge](https://sites.google.com/view/loveucvpr21/home)
- We use `eval.py` for evaluation in our competition.
- Although one can use frame_index and number of total frames to measure the *Rel.Dis*, as we implemented [here](https://github.com/StanLei52/GEBD/blob/main/eval/eval_GEBD_k400.ipynb), you should **represent the detected boundaries with timestamps (in seconds)**. For example:
  ```shell
  {
  ‘6Tz5xfnFl4c’: [5.9, 9.4], # boundaries detected at 5.9s, 9.4s of this video
  ‘zJki61RMxcg’: [0.6, 1.5, 2.7] # boundaries detected at 0.6s, 1.5s, 2.7s of this video
  ...
  }
  ```
- Refer to [this file](https://github.com/StanLei52/GEBD/blob/main/data/export/prepare_k400_release.ipynb) to generate GT files from raw annotation.
