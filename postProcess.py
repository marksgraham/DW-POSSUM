#!/usr/bin/env python

#Code to generate a full shell of diffusion-weighted, eddy distorted images using FSL's possum, along with data that can be used to
#establish a ground truth.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Handle arguments (before imports so --help can be fast)
def str2bool(v):
	#Function allows boolean arguments to take a wider variety of inputs
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
parser = argparse.ArgumentParser(description="Tidy up the simulations.")
parser.add_argument("simulation_dir",help="Path to the simulation directory (output_dir of generateFileStructure.py)")
parser.add_argument("num_images",help='Number of volumes.',type=int)
parser.add_argument("--simulate_artefact_free",help='Run simulation on datasets without eddy-current and motion artefacts. Default=True.', type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("--simulate_distorted",help='Run simulation datasets with eddy-current and motion artefacts. Default=False',type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("--noise_levels",help="Set sigma for the noise level in the dataset. Can pass multiple values seperated by spaces.",nargs="+",type=float)
parser.add_argument("--interleave_factor",help="Set this if the simulation slice order has been interleaved.",type=int,default=1)
parser.add_argument("--signal_dropout",help="Set this to simulate signal dropout.",type=str2bool,nargs='?',const=True,default=False)
args=parser.parse_args()

#Imports
import os
from subprocess import call
import sys
import numpy as np
from Library import possumLib as pl

#Assign args
simDir = os.path.abspath(args.simulation_dir)
numImages = args.num_images
normalImages = args.simulate_artefact_free
motionAndEddyImages = args.simulate_distorted
if args.noise_levels == None:
	noiseLevel = [0.0]
else:
	noiseLevel = args.noise_levels
print(noiseLevel)
interleaveFactor = args.interleave_factor


def saveImage(simDir,saveImageDir,fileName):
	call(["mv", simDir + "/image_abs.nii.gz", os.path.join(saveImageDir,fileName)])

def saveNoiseyImage(simDir,saveImageDir,fileName):
	call(["mv", simDir + "/imageNoise_abs.nii.gz", os.path.join(saveImageDir,fileName)])

def convertImageToFloat(imageDir, fileName):
	pathToImage = os.path.join(imageDir, fileName)
	call(["fslmaths", pathToImage, pathToImage, "-odt", "float"])

def readSignal(signalPath):
	fid = open(signalPath,"rb")
	testval= np.fromfile(fid, dtype=np.uint32,count = 1)
	dummy=np.fromfile(fid, dtype=np.uint32,count = 1)
	nrows=np.fromfile(fid, dtype=np.uint32,count = 1)[0]
	ncols=np.fromfile(fid, dtype=np.uint32,count = 1)[0]
	signal=np.fromfile(fid,dtype=np.float64,count = nrows*ncols)
	signal = np.reshape(signal,(nrows, ncols),order='F')
	return signal

def writeSignal(fname,mat):
    mvals = mat

    fidin = open(fname,"w")  
    magicnumber=42
    dummy=0
    [nrows,ncols]=mat.shape
    header = np.array([magicnumber,dummy,nrows,ncols])
    header.astype(np.uint32).tofile(fidin)
    mvals = np.reshape(mvals,[1,ncols*2],order='F')
    mvals.astype(np.float64).tofile(fidin)
    fidin.close()

def unInterleaveSignal(signal, numSlices, interleaveFactor):
	[nrows,ncols]=signal.shape
	signalUninterleaved = np.zeros((nrows,ncols))
	counter = 0
	entriesPerSlice = int(ncols/numSlices)
	for i in range(interleaveFactor):
		for j in range(i,numSlices,interleaveFactor):
			startIndexOld = counter* entriesPerSlice
			endIndexOld = (counter + 1) * entriesPerSlice -1
			startIndex = j* entriesPerSlice
			endIndex = (j + 1) * entriesPerSlice -1
			signalUninterleaved[:,startIndex:endIndex] = signal[:,startIndexOld:endIndexOld]
			counter = counter + 1
	return signalUninterleaved


resultsDir = simDir+"/Results"

for direction in range(numImages):
	if motionAndEddyImages == True:
		simDirDirectionMotionAndEddy = simDir+"/DirectionMotionAndEddy"+str(direction)
		if interleaveFactor > 1 or args.signal_dropout == True:
			signal = readSignal(simDirDirectionMotionAndEddy+'/signal')
			signalUninterleaved = unInterleaveSignal(signal,55,interleaveFactor)
			if args.signal_dropout == True:
				motion_level = pl.get_motion_level(simDirDirectionMotionAndEddy)
				signalUninterleaved = pl.add_signal_dropout(signalUninterleaved,motion_level,55,72*86)
			writeSignal(simDirDirectionMotionAndEddy+'/signalUninterleaved',signalUninterleaved)
		else:
			call(["cp",simDirDirectionMotionAndEddy+'/signal',simDirDirectionMotionAndEddy+'/signalUninterleaved'])

	if normalImages == True:
		simDirDirection = simDir+"/Direction"+str(direction)
		if interleaveFactor > 1:
			signal = readSignal(simDirDirection+'/signal')
			signalUninterleaved = unInterleaveSignal(signal,55,interleaveFactor)
			writeSignal(simDirDirection+'/signalUninterleaved',signalUninterleaved)
		else:
			call(["cp",simDirDirection+'/signal',simDirDirection+'/signalUninterleaved'])

	#Generate noise
	for sigma in noiseLevel:
		if normalImages == True:
			call(["systemnoise","-s",str(sigma),"-i",simDirDirection+"/signalUninterleaved","-o",simDirDirection+"/signalNoise"])
			call(["signal2image","-i",simDirDirection+"/signalNoise","-p",simDirDirection+"/pulse","-o",simDirDirection+"/imageNoise","-a"])


		if motionAndEddyImages == True:
			call(["systemnoise","-s",str(sigma),"-i",simDirDirectionMotionAndEddy+"/signalUninterleaved","-o",simDirDirectionMotionAndEddy+"/signalNoise"])
			call(["signal2image","-i",simDirDirectionMotionAndEddy+"/signalNoise","-p",simDirDirectionMotionAndEddy+"/pulse","-o",simDirDirectionMotionAndEddy+"/imageNoise","-a"])

		#Save
		if motionAndEddyImages == True:
			saveNoiseyImage(simDirDirectionMotionAndEddy,resultsDir,"diff+eddy+motion_sigma{}_image{}.nii.gz".format(sigma,direction))
			convertImageToFloat(resultsDir,"diff+eddy+motion_sigma{}_image{}.nii.gz".format(sigma,direction))
		if normalImages == True:
			saveNoiseyImage(simDirDirection,resultsDir,"diff_sigma{}_image{}.nii.gz".format(sigma,direction))
			convertImageToFloat(resultsDir,"diff_sigma{}_image{}.nii.gz".format(sigma,direction))

#Merge
if motionAndEddyImages == True:
	for sigma in noiseLevel:
		callMergeNoise = "fslmerge -t " + resultsDir + "/diff+eddy+motion_sigma{} ".format(sigma)
		callDelNoise = "rm "
		for i in range(numImages):
			callMergeNoise += resultsDir + "/diff+eddy+motion_sigma{}_image{}.nii.gz ".format(sigma,i)
			callDelNoise += resultsDir + "/diff+eddy+motion_sigma{}_image{}.nii.gz ".format(sigma,i)
		os.system(callMergeNoise)
		os.system(callDelNoise)


if normalImages == True:
	for sigma in noiseLevel:
		callMergeNoise = "fslmerge -t " + resultsDir + "/diff_sigma{} ".format(sigma)
		callDelNoise = "rm "
		for i in range(numImages):
			callMergeNoise += resultsDir + "/diff_sigma{}_image{}.nii.gz ".format(sigma,i)
			callDelNoise += resultsDir + "/diff_sigma{}_image{}.nii.gz ".format(sigma,i)
		os.system(callMergeNoise)
		os.system(callDelNoise)
