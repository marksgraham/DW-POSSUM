#!/usr/bin/env python
#Randomly generate motion files

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
from interpmot import possum_interpmot

def read_args():
	parser = argparse.ArgumentParser(description="Create semi-realistic motion files.")
	parser.add_argument("possum_dir",help="Path to possum directory the motion file will be used with.")
	parser.add_argument("output_dir",help="Path to output directory")
	parser.add_argument("num_volumes",help="Number of volumes to create motion trace for",type=int)
	parser.add_argument("motion_type",help="0 for continuous; 1 for between slices ; 2 for between volumes",type=int)
	parser.add_argument("drift_severity",help="set from 0-1: 0 is no drift, 1 corresponds to max drift of 5mm and 5deg.",type=float)
	parser.add_argument("drift_type",help="linear or cosine")
	parser.add_argument("spike_severity",help="set from 0-1: 1 corresponds to max spike of 5mm and 5deg.",type=float)
	parser.add_argument("spike_frequency",help="set from 0-1: sets probability of a motion spike in a volume",type=float)
	args=parser.parse_args()
	print(args.drift_type)
	if (args.drift_type != "linear" and args.drift_type != "cosine"): 
		raise ValueError('Slow drift motion must be cosine or linear')
	return args

def read_possum_params(possum_dir):
	pulse_info = np.loadtxt(os.path.join(possum_dir,'pulse.info'))
	TR = pulse_info[2]
	TRslice = pulse_info[3]
	num_slices = int(pulse_info[12])
	return TR, TRslice, num_slices

def initialise_motion_mat(TR,num_slices,num_volumes):
	max_time = TR * (num_volumes +1) #Extra volume to ensure motion_mat is long enough 
	time_increments = 0.1
	time = np.arange(0,max_time,time_increments)
	motion_mat = np.zeros((time.shape[0],7))
	motion_mat[:,0] = time
	return motion_mat

#Add slow-drift motion of size described by the command line
def add_slow_drift(motion_mat,severity,type,TR):
	time = motion_mat[:,0]
	max_translation = 5 * severity / 1000 #Max translations in m
	max_rotation = np.radians(5 * severity)  #Max rotation in rad
	if type == 'cosine':	
		#Add translations
		for i in range(1,4):
			period = np.random.normal(15, 5) * TR
			motion_mat[:,i] = motion_mat[:,i] \
			+ max_translation * random.uniform(-1,1) * np.cos(time/period)
		#Add rotations
		for i in range(4,7):
			period = np.random.normal(15, 5) * TR
			motion_mat[:,i] = motion_mat[:,i] \
			+ max_rotation * random.uniform(-1,1) * np.cos(time/period)
	elif type == 'linear':
		#Add translations
		for i in range(1,4):
			motion_mat[:,i] = motion_mat[:,i] \
			+  random.uniform(-1,1) * np.linspace(0,max_translation,time.shape[0])
		#Add rotations
		for i in range(4,7):
			motion_mat[:,i] = motion_mat[:,i] \
			+  random.uniform(-1,1) * np.linspace(0,max_rotation,time.shape[0])

	return motion_mat

#Add motion spikes with a severity/frequency described by the command line
def add_motion_spikes(motion_mat,frequency,severity,TR):
	time = motion_mat[:,0]
	max_translation = 5 * severity / 1000 * np.sqrt(2*np.pi)#Max translations in m, factor of sqrt(2*pi) accounts for normalisation factor in norm.pdf later on
	max_rotation = np.radians(5 * severity) *np.sqrt(2*np.pi) #Max rotation in rad
	time_blocks = np.floor(time[-1]/TR).astype(np.int32)
	for i in range(time_blocks):
		if np.random.uniform(0,1) < frequency: #Decide whether to add spike
			for j in range(1,4):
				if np.random.uniform(0,1) < 1/6:
					motion_mat[:,j] = motion_mat[:,j] \
					+ max_translation * random.uniform(-1,1) \
					* norm.pdf(time,loc = (i+0.5)*TR,scale = TR/5)
			for j in range(4,7):
				if np.random.uniform(0,1) < 1/6:
					motion_mat[:,j] = motion_mat[:,j] \
					+ max_rotation * random.uniform(-1,1) \
					* norm.pdf(time,loc = (i+0.5 + np.random.uniform(-0.25,-.25))*TR,scale = TR/5)
	return motion_mat




def plot_motion(motion_mat):
	time = motion_mat[:,0]
	plt.figure(figsize=(15,5))
	plt.subplot(1,2,1)
	plt.plot(time,motion_mat[:,1]* 1000,label='x')
	plt.plot(time,motion_mat[:,2]* 1000,label='y')
	plt.plot(time,motion_mat[:,3]* 1000,label='z')
	plt.xlabel('Time / s')
	plt.ylabel('Translation / mm')
	plt.legend()
	plt.subplot(1,2,2)
	plt.plot(time,np.degrees(motion_mat[:,4]),label='x')
	plt.plot(time,np.degrees(motion_mat[:,5]),label='y')
	plt.plot(time,np.degrees(motion_mat[:,6]),label='z')
	plt.ylabel('Rotations / degrees')
	plt.xlabel('Time / s')
	plt.legend()
	plt.show()

def get_motion_for_volume(motion_mat,TR,index):
	time_increments = 0.1 #set in initalise matrix function
	num_entries = np.floor(TR/ time_increments).astype(np.int32)
	start_index = index*num_entries
	end_index = (index+1)*num_entries
	return motion_mat[start_index:end_index,:]

def makeFolder(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

#Interpolate using MJ's code

#Save

if __name__ == "__main__":
	args = read_args()
	TR, TRslice, num_slices = read_possum_params(args.possum_dir)
	motion_mat = initialise_motion_mat(TR,num_slices,args.num_volumes)
	motion_mat = add_slow_drift(motion_mat,args.drift_severity,args.drift_type,TR)
	motion_mat = add_motion_spikes(motion_mat,args.spike_frequency,args.spike_severity,TR)
	#Divide into volumes and feed each into MJ's interpmot script
	makeFolder(args.output_dir)
	for i in range(args.num_volumes):
		motion_mat_volume = get_motion_for_volume(motion_mat,TR,i)
		motion_mat_volume_interp = possum_interpmot(motion_mat_volume,args.motion_type, TR, TRslice,num_slices,1)
		out_name = os.path.join(args.output_dir,'motion'+str(i)+'.txt')
		np.savetxt(out_name,motion_mat_volume_interp)


	#plot_motion(motion_mat)