import numpy as np
from subprocess import call
import os,sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=str)

settings = parser.parse_args(); 

param_fn = "params_waveform_save_%s.txt" % settings.expt
fo = open(param_fn, "w")


data_path = "/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/%s/data" % settings.expt
logs_path = "/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/%s/logs" % settings.expt

if not os.path.exists(data_path):
    os.makedirs(data_path)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

expt = settings.expt

print('testing!')


#mice = ['gmou03','gmou04','gmou05']
#mice = ['gmou11','gmou12']
#mice = ['gmou15']
#mice = ['gmou27']
#mice = ['gmou31','gmou32']
#mice = ['gmou31']
mice = ['gmou32']
###



i = 1
for mouse in mice:

    data_path = '/n/groups/datta/guitchounts/data/%s/' % mouse
    
    # take only folders within each mouse's dir that 1) start w/ 2020- and 2) have an 'experiment1' folder/file inside (that's where ephys data is). 
    #all_mouse_files = [fil for fil in os.listdir(data_path) if (os.path.isdir(data_path + fil) and fil.startswith('2020') and 'experiment1' in os.listdir(data_path + fil)  ) ] 
    #all_mouse_files = [fil for fil in os.listdir(data_path) if (os.path.isdir(data_path + fil) and mouse in fil ) ] 
    #all_mouse_files = ['gmou15_2021-07-26_17-11-53_odor']
    #all_mouse_files = ['gmou27_2021-08-26_10-10-19_odor']
    #all_mouse_files = ['gmou15_2021-08-20_10-58-01_odor','gmou15_2021-09-07_11-29-58_odor','gmou15_2021-04-07_11-05-36','gmou15_2021-08-31_11-07-55_odor','gmou15_2021-04-14_13-07-11','gmou15_2021-08-27_11-07-41_odor','gmou15_2021-04-02_13-31-50','gmou15_2021-04-13_14-30-18','gmou15_2021-08-20_15-30-30_odor','gmou15_2021-08-19_14-57-13_odor','gmou15_2021-07-26_13-22-20','gmou15_2021-04-12_14-36-34','gmou15_2021-04-01_12-38-44','gmou15_2021-08-30_12-02-20_odor','gmou15_2021-04-09_14-36-53','gmou15_2021-09-03_11-00-04_odor','gmou15_2021-09-02_09-33-07_odor','gmou15_2021-04-02_16-19-01','gmou15_2021-08-17_16-55-55_wheeltest','gmou15_2021-03-31_11-49-34','gmou15_2021-03-30_14-43-31','gmou15_2021-08-26_11-22-38_odor','gmou15_2021-04-08_12-06-19']

    all_mouse_files = [fil for fil in os.listdir(data_path) if (os.path.isdir(data_path + fil)   ) ] 
    #all_mouse_files = ['gmou31_2021-10-05_13-55-04_bucket', 'gmou31_2021-10-06_14-04-04_odor', 'gmou31_2021-10-07_12-24-05_bucket',
  #'gmou31_2021-10-05_11-51-37_odor', 'gmou31_2021-10-06_12-18-52_bucket', 'gmou31_2021-10-07_10-43-22_odor']

    for fil in all_mouse_files:

        base_path = '/n/groups/datta/guitchounts/data/%s/%s/' % (mouse,fil)

        print(mouse,fil,base_path ) 

        fo.write("-mouse %s -fil %s -base_path %s \n" % (mouse,fil,base_path) )

        i+=1

                    
fo.close()

call("python run_o2_array.py -cmd run_sorting.py -expt %s -cores 8 -hours 6 -mem 32000 -partition short -paramfile %s" % (expt,param_fn), shell=True)
