#!/usr/bin/env python
#Code to generate distorted pulse files only a full shell of diffusion-weighted, eddy distorted images using FSL's possum.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from subprocess import call
from dipy.io import read_bvals_bvecs
import dipy.reconst.shm as shm
import nibabel as nib
import numpy as np
import scipy.io
import argparse
import shutil

#Add root to pythonpath for lib import
dir_path = os.path.dirname(os.path.realpath(__file__))
package_path = os.path.abspath(os.path.join(dir_path,os.pardir))
sys.path.insert(0,package_path)
from Library import possumLib as pl

#Parse arguments
def str2bool(v):
	#Function allows boolean arguments to take a wider variety of inputs
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Setup all the files required to run the simulations.")

parser.add_argument("possum_dir",help="Path to the possum simulation directory.")
parser.add_argument("output_dir",help="Path to output directory.")
parser.add_argument("bvals",help="Path to bval file.")
parser.add_argument("bvecs",help="Path to bvec file.")
parser.add_argument("--num_images",help='Number of volumes. Defaults to number of entries in bval file.',type=int)
parser.add_argument("--interleave_factor",help="Interleave factor for slice-acquistion. Default=1. ",type=int,default=1)


args=parser.parse_args()

#Check arguments and setup paths
simDir = os.path.abspath(args.possum_dir)
outputDir = args.output_dir
bvalDir = args.bvals
bvecDir = args.bvecs
#Check bval entries
bvals, bvecs = read_bvals_bvecs(
		args.bvals, 
		args.bvecs)

#Check num_images
num_bvals = len(bvals)
if args.num_images:
	numImages= args.num_images
	if numImages > num_bvals:
		print("Warning: num_images cannot be greater than the number of entries in the bval file. Setting to this maximum.")
		numImages = num_bvals
else:
	numImages= num_bvals

interleaveFactor = args.interleave_factor


#Set eddy current distortion parameters
ep =0.006 #default is 0.006
tau = 0.1
basicSettings = [0]*4
basicSettings[0]=0.001  #Leave a short gap before first gradient pulse /s
basicSettings[1]=0.010   #Pulse width / s
basicSettings[2]=0.025   #Diffusion time/s
basicSettings[3]=0.06   #Diffusion gradient strength T/m
eddyGradients = 'decaying'; #Flat or decaying



#Set directories
codeDir =  os.path.abspath('.')

print('Output directory: ' + outputDir)


#Make directory for cluster files
pl.makeFolder(outputDir)



for index, bvec in enumerate(bvecs[0:numImages]):
	#This workaround lets you carry on if generating is interrupted
	if index < 0:
		pass
	else:
		#Read in pulse
		if index == 0:
			pulse=pl.read_pulse(simDir+"/pulse")
			pulseinfo = np.loadtxt(simDir+'/pulse.info')

		#Make distorted eddy pulse
		if eddyGradients=='flat':
			pl.generateEddyPulseFromBvecFlat(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)
		else:
			new_pulse = pl.addEddyAccordingToBvec(pulse,pulseinfo,basicSettings[0],basicSettings[1],basicSettings[2],basicSettings[3],ep, tau,bvals[index], bvec)

		#Interleave 
		if (interleaveFactor != 1):
			new_pulse = pl.interleavePulse(new_pulse,int(pulseinfo[12]),interleaveFactor)	

		#Save to correct location
		pl.write_pulse(outputDir+"/pulse"+str(index),new_pulse)


