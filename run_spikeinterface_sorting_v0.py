#!/usr/bin/env python
# coding: utf-8

import spikeinterface
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from glob import glob
import argparse
import multiprocessing
#get_ipython().run_line_magic('matplotlib', 'notebook')
#get_ipython().run_line_magic('matplotlib', 'inline')

parser = argparse.ArgumentParser()

#parser.add_argument('-paramfile', type=argparse.FileType('r'))
#parser.add_argument('-line', type=int)


parser.add_argument('-base_path', type=str, help="main directory containing sessions for each mouse")
#parser.add_argument('-mouse', type=str)
parser.add_argument('-fil', type=str,help='directory containing mkv and ephys files')
parser.add_argument('-prbfile', type=str)

settings = parser.parse_args(); 

base_path = settings.base_path #'/n/groups/datta/maya/ofa-snr/'
#mouse = settings.mouse
fil = settings.fil
prb_file = settings.prbfile
session_dir = os.path.join(base_path,fil)




#prb_file = recording_dir+'A1x32-5mm-50-177-A32.prb'
prb_file = os.path.join(base_path,prb_file)


recording = se.OpenEphysRecordingExtractor(session_dir)
fs = recording.get_sampling_frequency()
duration = recording.get_num_frames()

recording_prb = recording.load_probe_file(probe_file=prb_file)



print(fs,duration)
print(duration/fs/3600) #how many hours



w_el_circ = sw.plot_electrode_geometry(recording_prb)



#Preprocess Bandpass
recording_f = st.preprocessing.bandpass_filter(recording_prb, freq_min=300, freq_max=6000)

recording_rm_noise = st.preprocessing.remove_bad_channels(recording_f, bad_channel_ids=[])

#Re-referencing.
recording_cmr = st.preprocessing.common_reference(recording_rm_noise, reference='median')


#Should see some spikies!
plt_kwargs = {'linewidth':0}
fig, ax = plt.subplots(1,1, figsize=(10,10),**plt_kwargs)
sw.plot_timeseries(recording_cmr, trange=[11,16], ax=ax,figure=fig)

plt.savefig(os.path.join(session_dir,'raw_filtered_timeseries.pdf'),format='pdf')



print('Installed sorters', ss.installed_sorters())
#print(ss.get_default_params('mountainsort4'))



default_ms4_params = ss.get_default_params('mountainsort4')
ms4_params = default_ms4_params.copy()
ms4_params['adjacency_radius'] = 100
ms4_params['detect_sign'] = 0
ms4_params['filter'] = False

num_workers = 8
ms4_params['num_workers'] = num_workers
ms4_params['curation'] = False

ms4_params['detect_interval'] = 30 ### 1 ms between spikes

print(ms4_params)

print('Starting mountainsort4 sorting....')

start = time.time()
print("Output folder=" + session_dir+'/tmp_MS4_sort')
#don't run parallel=True if just running one session. 
sorting_MS4_2 = ss.run_mountainsort4(recording=recording_cmr, parallel=False,
                                     verbose=True,
                                     output_folder = session_dir+'/tmp_MS4_sort', **ms4_params)

print('sorting finished in ',time.time() - start )



print('Units found by Mountainsort4:', sorting_MS4_2.get_unit_ids())


sorting_MS4_2.dump_to_pickle(session_dir+'/tmp_MS4/sorting.pkl')


metrics = st.validation.compute_quality_metrics(sorting=sorting_MS4_2, recording=recording_cmr,
                                                metric_names=['firing_rate', 'isi_violation', 'snr' ],
                                                as_dataframe=True)


good_units = (np.intersect1d(np.where(metrics.isi_violation < 0.01)[0],
                                       np.where(metrics.snr > 4)[0])#,np.where(snrs > 5)[0]
                      )


waveform_params = st.postprocessing.get_waveforms_params()
waveform_params['ms_before'] = 1
waveform_params['ms_after'] = 1
#waveform_params['max_channels_per_waveforms'] = 1 
#waveform_params['n_jobs'] = 8
waveform_params['max_spikes_per_unit'] = 1000

waveform_params['compute_property_from_recording'] = False

start = time.time()

waveforms = st.postprocessing.get_unit_waveforms(recording_cmr,sorting_MS4_2, return_idxs=False,
                                                recompute_info=True,
                                                 #unit_ids=good_units,
                                        save_as_features=True, 
                                                 verbose=True,
                                                 **waveform_params
                                                )
print(time.time() - start)



np.savez(session_dir+'/tmp_MS4/waveforms_all.npz',{item : waveforms[item] for item in range(len(waveforms))})



np.savez(session_dir+'/tmp_MS4/unit_properties.npz' ,
         isi_violations=metrics.isi_violation.values,
         frs=metrics.firing_rate.values,
         snrs=metrics.snr.values,good_units=good_units)



data_channels = {}

for i in range(len(recording_cmr.get_channel_ids())):
    data_channels[recording_cmr.get_channel_ids()[i]] = recording_cmr.get_channel_locations()[i]



data_channels.keys()


#Find best channel for each unit to plot an accurate,representative waveform
best_ch_per_unit = {}
for u in np.arange(np.array(waveforms).shape[0]):
    dev = []
    for ch in np.arange(waveforms[0].shape[1]):
        #dev.append(waveforms[u][:,ch,:].mean(axis=0).std())#high std means big flux on this channel
        dev.append(np.abs(waveforms[u][:,ch,:].mean(axis=0)).max())
        #best_ch_per_unit[u] = [list(data_channels.keys())[i] for i in np.array(dev).argsort()[-3:][::-1]] #top 3 channels
        best_ch_per_unit[u] = list(data_channels.keys())[np.argmax(dev)] #single best channel


#All waveforms are strictly taken from the one best channel for each unit [ALL SORTED UNITS]
fig,ax = plt.subplots(10,int(np.round(len(waveforms)/10)+1), figsize=(15,8), sharex=True)

for _ax,i in zip(ax.ravel(),range(len(waveforms))):
    _ax.plot(waveforms[i]
              [:,list(data_channels.keys()).index(best_ch_per_unit[i])].T ,c='#12100E',lw=0.25,alpha=0.5)

    _ax.plot(waveforms[i][:,list(data_channels.keys()).index(best_ch_per_unit[i])].mean(axis=(0)),c='#FCA311',lw=2)
    
    #_ax.set_title("Unit"+str(i))
    #_ax.set_title(str(i)+":"+ str(np.round(snrs[i],2)))
    #_ax.set_title("#"+str(good_units[i])+":SNR "+str(np.round(metrics.snr.iloc[good_units[i]],2))+",FR:"+str(np.round(metrics.firing_rate.iloc[i],2)))
    _ax.set_title("#"+str(i)+':'+str(i in good_units))
    #_ax.set_ylim([-10,10])
    
plt.savefig(os.path.join(session_dir,'all_unit_waveforms.pdf'),format='pdf')




#All waveforms are strictly taken from the one best channel for each unit [GOOD UNITS ONLY]
fig,ax = plt.subplots(3,int(np.round(len(good_units)/3)+1), figsize=(15,8), sharex=True)

for _ax,i in zip(ax.ravel(),range(len(good_units))):
    _ax.plot(waveforms[good_units[i]]
              [:,list(data_channels.keys()).index(best_ch_per_unit[good_units[i]])].T ,c='#12100E',lw=0.25,alpha=0.5)

    _ax.plot(waveforms[good_units[i]][:,list(data_channels.keys()).index(best_ch_per_unit[good_units[i]])].mean(axis=(0)),c='#FCA311',lw=2)
    
    #_ax.set_title("Unit"+str(i))
    #_ax.set_title(str(i)+":"+ str(np.round(snrs[i],2)))
    _ax.set_title("#"+str(good_units[i])+":SNR "+str(np.round(metrics.snr.iloc[good_units[i]],2))+",FR:"+str(np.round(metrics.firing_rate.iloc[i],2)))
    #_ax.set_title("#"+str(good_units[i])+'channel:'+str(best_ch_per_unit[good_units[i]]))
    #_ax.set_ylim([-10,10])
    
plt.savefig(os.path.join(session_dir,'good_unit_waveforms.pdf'),format='pdf')


#good unit raster
sw.plot_rasters(sorting_MS4_2,trange=[15,20], unit_ids = good_units)
plt.savefig(os.path.join(session_dir,'good_unit_raster.pdf'),format='pdf')


#all unit ISI dists
fig,ax = plt.subplots(figsize=(10,10))
sw.plot_isi_distribution(sorting_MS4_2, bins=50,window=1,figure=fig, ax=ax)
plt.savefig(os.path.join(session_dir,'all_unit_isi_dist.pdf'),format='pdf')



#Should be constant through session
fig,ax = plt.subplots(figsize=(10,10))
sw.plot_amplitudes_timeseries(recording_cmr, sorting_MS4_2, max_spikes_per_unit=100, unit_ids = good_units, ax=ax)
plt.savefig(os.path.join(session_dir,'good_unit_amplitude_timeseries.pdf'),format='pdf')


#good unit autocorrs
fig,ax = plt.subplots(figsize=(10,10))
sw.plot_autocorrelograms(sorting_MS4_2, bin_size=0.01, window=0.4, unit_ids=good_units,  ax=ax)
plt.savefig(os.path.join(session_dir,'good_unit_autocorr.pdf'),format='pdf')


#If very asymmetrical, you may be over-splitting spikes, good unit crosscorr
fig,ax = plt.subplots(figsize=(10,10))
sw.plot_crosscorrelograms(sorting_MS4_2, bin_size=0.01, window=0.4, unit_ids=good_units,  ax=ax)
plt.savefig(os.path.join(session_dir,'good_unit_crosscorr.pdf'),format='pdf')





