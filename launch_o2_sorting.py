import numpy as np
from subprocess import call
import os,sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=str)

settings = parser.parse_args(); 

param_fn = "params_waveform_save_%s.txt" % settings.expt
fo = open(param_fn, "w")

os.makedirs("/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/%s/data" % settings.expt)
os.makedirs("/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/%s/logs" % settings.expt) # /n/home13/asaxe/gendynamics/results/randrelu

expt = settings.expt



mice = ['gmou03','gmou04','gmou05']

###



i = 1
for mouse in mice:

    data_path = '/n/groups/datta/guitchounts/data/%s/' % mouse
    
    all_mouse_files = [fil for fil in os.listdir(data_path) if (os.path.isdir(data_path + fil) and fil.startswith('2020') ) ] 

    for fil in all_mouse_files:

        base_path = '/n/groups/datta/guitchounts/data/%s/%s/' % (mouse,fil)

        print(mouse,fil,base_path ) 

        fo.write("-mouse %s -fil %s -base_path %s \n" % (mouse,fil,base_path) )

        i+=1

                    
fo.close()

call("python run_o2_array.py -cmd run_sorting.py -expt %s -cores 8 -hours 6 -mem 64000 -partition short -paramfile %s" % (expt,param_fn), shell=True)
