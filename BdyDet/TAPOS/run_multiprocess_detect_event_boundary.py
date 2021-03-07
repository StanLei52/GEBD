from multiprocessing import Pool 
import os
import time
import random

num_processes = 75 #set according to your machine
input_dir = '../data/exp_TAPOS/_featdiff_rgb/pred_err/'

vid_list = os.listdir(input_dir)
# e.g. err_vid_2IO8DO_QbRE_s00018_5_324_13_160_currentframe_000000.pkl
for i in range(len(vid_list)):
    vid_list[i]=vid_list[i][8:-24]
vid_set = set(vid_list)

def run_process(param):
    cmd = "python detect_event_boundary.py '"+str(param)+"'"
    time.sleep(random.random()*20)
    os.system(cmd)
    
    
pool = Pool(processes=num_processes)                                                        
pool.map(run_process, tuple(list(vid_set))) 
