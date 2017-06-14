#!/usr/bin/env python

#Code to generate a full shell of diffusion-weighted, eddy distorted images using FSL's possum, along with data that can be used to
#establish a ground truth.

import os
from subprocess import call
import sys
import numpy as np
import argparse
#import possumLib as pl
#pl=reload(pl)

parser = argparse.ArgumentParser(description="Tidy up the simulations.")

parser.add_argument("simulation_dir",help="Path to the simulation directory (output_dir of generateFileStructure.py)")
parser.add_argument("num_images",help='Number of volumes.',type=int)
parser.add_argument("--simulate_artefact_free",help='Run simulation on datasets without eddy-current and motion artefacts. Default=True.',action="store_true",default=1)
parser.add_argument("--simulate_distorted",help='Run simulation datasets with eddy-current and motion artefacts. Default=False',action="store_true",default=0)
parser.add_argument("--noise_levels",help="Set sigma for the noise level in the dataset. Can pass multiple values seperated by spaces.",nargs="+",default=0,type=float)
parser.add_argument("--interleave_factor",help="Set this if the simulation slice order has been interleaved.",type=int,default=1)

args=parser.parse_args()

simDirCluster = os.path.abspath(args.simulation_dir)
numImages = args.num_images
normalImages = args.simulate_artefact_free
motionAndEddyImages = args.simulate_distorted

print args.noise_levels

noiseLevel = args.noise_levels

for sigma in noiseLevel:
	print sigma

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
	nrows=np.fromfile(fid, dtype=np.uint32,count = 1)
	ncols=np.fromfile(fid, dtype=np.uint32,count = 1)
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




resultsDir = simDirCluster+"/Results"

for direction in range(numImages):
	if motionAndEddyImages == True:
		simDirClusterDirectionMotionAndEddy = simDirCluster+"/DirectionMotionAndEddy"+str(direction)
		if interleaveFactor > 1:
			signal = readSignal(simDirClusterDirectionMotionAndEddy+'/signal')
			signalUninterleaved = unInterleaveSignal(signal,55,interleaveFactor)
			writeSignal(simDirClusterDirectionMotionAndEddy+'/signalUninterleaved',signalUninterleaved)
		else:
			call(["cp",simDirClusterDirectionMotionAndEddy+'/signal',simDirClusterDirectionMotionAndEddy+'/signalUninterleaved'])

	if normalImages == True:
		simDirClusterDirection = simDirCluster+"/Direction"+str(direction)
		if interleaveFactor > 1:
			signal = readSignal(simDirClusterDirection+'/signal')
			signalUninterleaved = unInterleaveSignal(signal,55,interleaveFactor)
			writeSignal(simDirClusterDirection+'/signalUninterleaved',signalUninterleaved)
		else:
			call(["cp",simDirClusterDirection+'/signal',simDirClusterDirection+'/signalUninterleaved'])

	#Generate noise
	for sigma in noiseLevel:
		if normalImages == True:
			call(["systemnoise","-s",str(sigma),"-i",simDirClusterDirection+"/signalUninterleaved","-o",simDirClusterDirection+"/signalNoise"])
			call(["signal2image","-i",simDirClusterDirection+"/signalNoise","-p",simDirClusterDirection+"/pulse","-o",simDirClusterDirection+"/imageNoise","-a"])


		if motionAndEddyImages == True:
			call(["systemnoise","-s",str(sigma),"-i",simDirClusterDirectionMotionAndEddy+"/signalUninterleaved","-o",simDirClusterDirectionMotionAndEddy+"/signalNoise"])
			call(["signal2image","-i",simDirClusterDirectionMotionAndEddy+"/signalNoise","-p",simDirClusterDirectionMotionAndEddy+"/pulse","-o",simDirClusterDirectionMotionAndEddy+"/imageNoise","-a"])

		#Save
		if motionAndEddyImages == True:
			saveNoiseyImage(simDirClusterDirectionMotionAndEddy,resultsDir,"diff+eddy+motion_sigma{}_image{}.nii.gz".format(sigma,direction))
			convertImageToFloat(resultsDir,"diff+eddy+motion_sigma{}_image{}.nii.gz".format(sigma,direction))
		if normalImages == True:
			saveNoiseyImage(simDirClusterDirection,resultsDir,"diff_sigma{}_image{}.nii.gz".format(sigma,direction))
			convertImageToFloat(resultsDir,"diff_sigma{}_image{}.nii.gz".format(sigma,direction))

#Merge
if motionAndEddyImages == True:
	for sigma in noiseLevel:
		callMergeNoise = "fslmerge -a " + resultsDir + "/diff+eddy+motion_sigma{} ".format(sigma)
		callDelNoise = "rm "
		for i in range(numImages):
			callMergeNoise += resultsDir + "/diff+eddy+motion_sigma{}_image{}.nii.gz ".format(sigma,i)
			callDelNoise += resultsDir + "/diff+eddy+motion_sigma{}_image{}.nii.gz ".format(sigma,i)
		os.system(callMergeNoise)
		os.system(callDelNoise)


if normalImages == True:
	for sigma in noiseLevel:
		callMergeNoise = "fslmerge -a " + resultsDir + "/diff_sigma{} ".format(sigma)
		callDelNoise = "rm "
		for i in range(numImages):
			callMergeNoise += resultsDir + "/diff_sigma{}_image{}.nii.gz ".format(sigma,i)
			callDelNoise += resultsDir + "/diff_sigma{}_image{}.nii.gz ".format(sigma,i)
		os.system(callMergeNoise)
		os.system(callDelNoise)
