import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt

import pickle
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
import h5py


import matplotlib
matplotlib.use('pdf')

plt.rcParams['pdf.fonttype'] = 'truetype'

from functools import reduce
import datetime

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

import argparse
from subprocess import call
import time

from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument('-paramfile', type=argparse.FileType('r'))
parser.add_argument('-line', type=int)


parser.add_argument('-base_path', type=str)
parser.add_argument('-mouse', type=str)
parser.add_argument('-fil', type=str)

settings = parser.parse_args(); 


# Read in parameters from correct line of file
if settings.paramfile is not None:
    for l, line in enumerate(settings.paramfile):
        if l == settings.line:
            settings = parser.parse_args(line.split())
            break
            


base_path = settings.base_path # =  '/n/groups/datta/guitchounts/data/%s/%s/' % (mouse,fil)
mouse = settings.mouse ## e.g. gmou03
fil = settings.fil ## e.g. 2020-07-20_16-21-09


print('Running Spike Interface script on %s path... Path exists==%s' % (base_path,os.path.exists(base_path)) )




data = se.OpenEphysRecordingExtractor(glob('%s/Record*' % base_path)[0])

#recording = data.load_probe_file('/home/gg121/code/spikeinterface_analysis/64ch_cnt_probe_mapped.prb')

probe_file = glob('%s/../*.prb' % base_path)[0]  #'/home/gg121/code/spikeinterface_analysis/A4x16-Poly3-5mm-20-200-160-H64LP.prb'

recording = data.load_probe_file(probe_file)

#recording = data.load_probe_file(probe_file='/n/groups/datta/guitchounts/data/64ch_cnt_probe_shanks.prb') # '%s/../../32_groups.prb' % base_path

print('Channel ids:', recording.get_channel_ids())
print('Loaded properties', recording.get_shared_channel_property_names())
#print('Label of channel 0:', recording.get_channel_property(channel_id=0, property_name='label'))


recording_f = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
recording_cmr = st.preprocessing.common_reference(recording_f, reference='median')


num_workers = 8


default_ms4_params = ss.Mountainsort4Sorter.default_params()
#print(default_ms4_params)
ms4_params = default_ms4_params.copy()
ms4_params['adjacency_radius'] = 50
ms4_params['detect_sign'] = 0
ms4_params['filter'] = False
ms4_params['num_workers'] = num_workers
fs = 3e4




# default_kl_params = ss.KlustaSorter.default_params()
# #print(default_kl_params)
# kl_params = default_kl_params.copy()
# kl_params['adjacency_radius'] = 50
# kl_params['extract_s_before'] = 32
# kl_params['detect_sign'] = 0
# #kl_params['filter'] = False
# #kl_params['num_workers'] = num_workers





#### run sorting:
print('Starting mountainsort4 sorting....')
start = time.time()
# run Mountainsort:
sorting_MS4 = ss.run_sorter('mountainsort4',recording_cmr, #grouping_property='group',
                       parallel=True,
                       output_folder='%s/tmp_MS4' % base_path, **ms4_params)
print('sorting finished in %f seconds...' % (time.time() - start) )


#sorting_MS4 = st.curation.threshold_firing_rates(sorting_MS4, 
#    duration_in_frames=recording_cmr.get_num_frames(), threshold=0.05, threshold_sign='less')

#sorting_MS4 = st.curation.threshold_snrs(sorting_MS4, recording_cmr, threshold=5, threshold_sign='less')

#st.postprocessing.export_to_phy(recording_cmr, sorting_MS4, output_folder='%s/phy_MS4' % base_path, grouping_property='group')

num_frames = recording_cmr.get_num_frames()

all_unit_ids = sorting_MS4.get_unit_ids()

# high_snr_units =  np.where(st.validation.compute_snrs(sorting_MS4, recording_cmr) > 3.5)[0] 
# isi_units = np.where(st.validation.compute_isi_violations(sorting_MS4, num_frames) < 1)[0]

metrics = st.validation.compute_quality_metrics(sorting=sorting_MS4, recording=recording_cmr,
                                                metric_names=['firing_rate', 'isi_violation', 'snr', 'nn_hit_rate', 'nn_miss_rate'],
                                                as_dataframe=True)


#good_units = np.intersect1d(high_snr_units,isi_units) # for plotting 
good_units = np.intersect1d(np.where(metrics['isi_violation'].values <= 1.5), np.where(metrics['snr'].values >=3.5 ) )

np.savez('%s/tmp_MS4/unit_properties' % base_path,all_unit_ids=all_unit_ids,metrics=metrics,good_units=good_units)

print('saved unit properties')

sorting_MS4.dump_to_pickle('%s/tmp_MS4/sorting.pickle' % base_path)
print('saved unit sorting pickle')

waveform_params = st.postprocessing.get_waveforms_params()
waveform_params['ms_before'] = 1
waveform_params['ms_after'] = 1
#waveform_params['max_channels_per_waveforms'] = 1 
waveform_params['n_jobs'] = 8
waveform_params['max_spikes_per_unit'] = 1000



waveforms = st.postprocessing.get_unit_waveforms(recording_cmr,sorting_MS4, return_idxs=False,
                                                 **waveform_params
                                                )



np.savez('%s/tmp_MS4/waveforms' % base_path,{item : waveforms[item] for item in range(len(waveforms))})


print('saved waveforms')


# print('Starting klusta sorting....')
# # run KlustaKwik:
# sorting_KL = ss.run_sorter('klusta',recording_cmr, #grouping_property='group',
#                     parallel=True,
#                        output_folder='%s/tmp_KL' % base_path, **kl_params)

# # sorting_KL = st.curation.threshold_firing_rates(sorting_KL, 
# #     duration_in_frames=recording_cmr.get_num_frames(), threshold=0.05, threshold_sign='less')

# # sorting_KL = st.curation.threshold_snrs(sorting_KL, recording_cmr, threshold=5, threshold_sign='less')

# # st.postprocessing.export_to_phy(recording_cmr, sorting_KL, output_folder='%s/phy_KL' % base_path, grouping_property='group')








# print('Number of KlustaKwik units after curation:', len(sorting_KL.get_unit_ids()))
# #print('Number of Mountainsort units after curation:', len(sorting_MS4.get_unit_ids()))




## plot 
#f = plt.figure(dpi=600)
# ax = plt.subplot(111)
# sw.plot_unit_waveforms(recording_cmr,sorting_MS4,ax=ax,unit_ids=sorting_MS4.get_unit_ids(),
#                       max_channels=1, ms_after=1,
#                               )

### get mapping of channel id to location:
channel_ids = recording.get_channel_ids()
channel_locations = recording.get_channel_locations()
data_channels = {}
for i in range(64):
    data_channels[channel_ids[i]] = channel_locations[i]


for unit in good_units:

    f = plt.figure(dpi=600,figsize=(4,4))
    for ch,key in enumerate(data_channels):
    #for key in range(64):
        plt.plot(np.arange(0,20,20/60) + data_channels[key][0],
                waveforms[unit][:, ch, 60:120].mean(axis=0) + data_channels[key][1], 
                color='k', lw=0.25)

    plt.title('unit %d, snr=%.2f, ISI=%.2f' % (unit,metrics['snr'].values[unit],
                                               metrics['isi_violation'].values[unit] ) )
    sns.despine(left=True,bottom=True)
    
    f.savefig('%s/tmp_MS4/unit_%d.pdf' % (base_path,unit) )
    
    plt.close(f)


#f.savefig('%s/tmp_MS4/waveforms.pdf' % base_path)

print('plotting waveforms')

# ## plot 
# f = plt.figure(dpi=600)
# ax = plt.subplot(111)
# sw.plot_unit_waveforms(recording_cmr,sorting_KL,ax=ax,unit_ids=sorting_KL.get_unit_ids(),
#                       max_channels=1, ms_after=1,
#                               )
# f.savefig('%s/tmp_KL/waveforms.pdf' % base_path)


