import pickle
import numpy as np

# load GT files
with open('PATH_TO_GT_FILES.pkl', 'rb') as f:
    gt_dict = pickle.load(f, encoding='lartin1')

# load output files
with open('PATH_TO_OUTPUT_FILES.pkl','rb') as f:
    pred_dict = pickle.load(f, encoding='lartin1')

# recall precision f1 for threshold 0.05(5%)
threshold = 0.05
tp_all = 0
num_pos_all = 0
num_det_all = 0

for vid_id in list(gt_dict.keys()):

    # filter by avg_f1 score
    if gt_dict[vid_id]['f1_consis_avg']<0.3:
        continue

    if vid_id not in pred_dict.keys():
        continue

    # detected timestamps
    bdy_timestamps_det = pred_dict[vid_id]


    myfps = gt_dict[vid_id]['fps']
    my_dur = gt_dict[vid_id]['video_duration']
    ins_start = 0
    ins_end = my_dur 

    # remove detected boundary outside the action instance
    tmp = []
    for det in bdy_timestamps_det:
        tmpdet = det + ins_start
        if tmpdet >= (ins_start) and tmpdet <= (ins_end):
            tmp.append(tmpdet)
    bdy_timestamps_det = tmp
    if bdy_timestamps_det == []:
        num_pos_all += len(gt_dict[vid_id]['substages_timestamps'][0])
        continue
    num_det = len(bdy_timestamps_det)
    num_det_all += num_det

    # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
    bdy_timestamps_list_gt_allraters = gt_dict[vid_id]['substages_timestamps']
    f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
    tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
    num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

    for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
        bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
        num_pos = len(bdy_timestamps_list_gt)
        tp = 0
        offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det))) 
        for ann1_idx in range(len(bdy_timestamps_list_gt)):
            for ann2_idx in range(len(bdy_timestamps_det)):
                offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_list_gt[ann1_idx]-bdy_timestamps_det[ann2_idx])
        for ann1_idx in range(len(bdy_timestamps_list_gt)):
            if offset_arr.shape[1] == 0:
                break
            min_idx = np.argmin(offset_arr[ann1_idx, :])
            if offset_arr[ann1_idx, min_idx] <= threshold*my_dur:
                tp += 1
                offset_arr = np.delete(offset_arr, min_idx, 1)   

        num_pos_tmplist[ann_idx] = num_pos
        fn = num_pos - tp
        fp = num_det - tp
        if num_pos == 0: 
            rec = 1
        else:
            rec = tp/(tp+fn)
        if (tp+fp) == 0: 
            prec = 0
        else: 
            prec = tp/(tp+fp)
        if (rec+prec) == 0:
            f1 = 0
        else:
            f1 = 2*rec*prec/(rec+prec)            
        tp_tmplist[ann_idx] = tp
        f1_tmplist[ann_idx] = f1

    ann_best = np.argmax(f1_tmplist)
    tp_all += tp_tmplist[ann_best]
    num_pos_all += num_pos_tmplist[ann_best]

fn_all = num_pos_all - tp_all
fp_all = num_det_all - tp_all
if num_pos_all == 0:
    rec = 1
else:
    rec = tp_all/(tp_all+fn_all)
if (tp_all+fp_all) == 0:
    prec = 0
else:
    prec = tp_all/(tp_all+fp_all)
if (rec+prec) == 0:
    f1 = 0
else:
    f1 = 2*rec*prec/(rec+prec)

print(f1)