import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
import h5py
import matplotlib.gridspec as gridspec
plt.rcParams['pdf.fonttype'] = 'truetype'
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
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



data = se.OpenEphysRecordingExtractor(base_path)

recording = data.load_probe_file(probe_file='%s/../../32_groups.prb' % base_path)

print('Channel ids:', recording.get_channel_ids())
print('Loaded properties', recording.get_shared_channel_property_names())
print('Label of channel 0:', recording.get_channel_property(channel_id=0, property_name='label'))

recording_f = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
recording_cmr = st.preprocessing.common_reference(recording_f, reference='median')


default_ms4_params = ss.Mountainsort4Sorter.default_params()
#print(default_ms4_params)
ms4_params = default_ms4_params.copy()
ms4_params['adjacency_radius'] = 0
ms4_params['detect_sign'] = 0
ms4_params['filter'] = False
ms4_params['num_workers'] = 8
fs = 3e4


default_kl_params = ss.KlustaSorter.default_params()
#print(default_kl_params)
kl_params = default_kl_params.copy()
kl_params['adjacency_radius'] = 0
kl_params['extract_s_before'] = 32
kl_params['detect_sign'] = 0
kl_params['filter'] = False
kl_params['num_workers'] = 8


#### run sorting:

# run Mountainsort:
sorting_MS4 = ss.run_sorter('mountainsort4',recording_cmr,grouping_property='group',
                       output_folder='%s/tmp_MS4' % base_path, **ms4_params)

sorting_MS4 = st.curation.threshold_firing_rates(sorting_MS4, 
    duration_in_frames=recording_cmr.get_num_frames(), threshold=0.05, threshold_sign='less')

sorting_MS4 = st.curation.threshold_snrs(sorting_MS4, recording_cmr, threshold=5, threshold_sign='less')

st.postprocessing.export_to_phy(recording_cmr, sorting_MS4, output_folder='%s/phy_MS4' % base_path, grouping_property='group')




# run KlustaKwik:
sorting_KL = ss.run_sorter('klusta',recording_cmr,grouping_property='group',
                       output_folder='%s/tmp_KL' % base_path, **kl_params)

sorting_KL = st.curation.threshold_firing_rates(sorting_KL, 
    duration_in_frames=recording_cmr.get_num_frames(), threshold=0.05, threshold_sign='less')

sorting_KL = st.curation.threshold_snrs(sorting_KL, recording_cmr, threshold=5, threshold_sign='less')

st.postprocessing.export_to_phy(recording_cmr, sorting_KL, output_folder='%s/phy_KL' % base_path, grouping_property='group')







print('Number of KlustaKwik units after curation:', len(sorting_KL.get_unit_ids()))
print('Number of Mountainsort units after curation:', len(sorting_MS4.get_unit_ids()))




## plot 
f = plt.figure(dpi=600)
ax = plt.subplot(111)
sw.plot_unit_waveforms(recording_cmr,sorting,ax=ax,unit_ids=sorting_MS4.get_unit_ids(),
                      max_channels=1, ms_after=1,
                              )
f.savefig('%s/tmp_MS4/waveforms.pdf' % base_path)


## plot 
f = plt.figure(dpi=600)
ax = plt.subplot(111)
sw.plot_unit_waveforms(recording_cmr,sorting,ax=ax,unit_ids=sorting_KL.get_unit_ids(),
                      max_channels=1, ms_after=1,
                              )
f.savefig('%s/tmp_KL/waveforms.pdf' % base_path)


