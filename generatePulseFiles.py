#!/usr/bin/env python

#Code to generate a full shell of diffusion-weighted, eddy distorted images using FSL's possum, along with data that can be used to
#establish a ground truth.

import os
from subprocess import call
from dipy.io import read_bvals_bvecs
import dipy.reconst.shm as shm
import nibabel as nib
import possumLib as pl
import numpy as np
import scipy.io
#pl = reload(pl)

################Check the following are correct before running#################
#Set relevant simulation directories:
simDir = os.path.abspath('Files/possumSimdirSESliceRes4SusceptExtendedFOV')

#set output directory
outputDirList = '../MotionWithSus/JesperPaper/pulseFilesExtendedFOV';

#Load in bvecs, bvals
bvalDirList = 'bvalsfmrib'
bvecDirList = 'bvecsfmrib'

#Choose number of images to generate (must be <= length of bval file)
numImagesList=65;


#Choose whether to keep artefact-free images
normalImages = "off";

#Choose whether to generate distorted images
motionAndEddyImages = "on";

###############################################################################



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
matlabDir = "/Applications/MATLAB_R2014b.app/bin/matlab"



bvals, bvecs = read_bvals_bvecs(
	'../Code/SimulateImages/bvalsbvecs/'+bvalDirList, 
	'../Code/SimulateImages/bvalsbvecs/'+bvecDirList)
print bvals
print bvecs



#Make directory for cluster files
simDirCluster=outputDirList
call(["mkdir",simDirCluster])
pl.initialise(simDir,codeDir)



for index, bvec in enumerate(bvecs[0:numImagesList]):
	#This workaround lets you carry on if generating is interrupted
	if index  == 0 :
		#Copy over all pulse files
		call(["cp",simDir+"/pulse.info",simDirCluster + '/'])
		call(["cp",simDir+"/pulse.readme",simDirCluster + '/'])
		call(["cp",simDir+"/pulse.posx",simDirCluster + '/'])
		call(["cp",simDir+"/pulse.posy",simDirCluster + '/'])
		call(["cp",simDir+"/pulse.posz",simDirCluster + '/'])
		call(["cp",simDir+"/pulse",simDirCluster + '/'])



		
	#Make distorted eddy pulse
	if eddyGradients=='flat':
		pl.generateEddyPulseFromBvecFlat(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)
	else:
		#pl.generateEddyPulseFromBvec(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)
		pl.addEddyAccordingToBvec(basicSettings[0],basicSettings[1],basicSettings[2],basicSettings[3],ep, tau,bvals[index], bvec)

	#Move eddy distorted pulse to simdir
	call(["mv",codeDir + "/pulse_new", simDirCluster+"/pulse"+str(index)])




#Tidy up:
pl.tidyUp(simDir,codeDir)

