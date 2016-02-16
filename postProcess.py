#!/usr/bin/env python

#Code to generate a full shell of diffusion-weighted, eddy distorted images using FSL's possum, along with data that can be used to
#establish a ground truth.

import os
from subprocess import call
import sys
import numpy as np
#import possumLib as pl
#pl=reload(pl)

###################################Check these before running######################################################
simDirCluster = os.path.abspath('../../SliceMotion/stepMotionMassive/')
numImages = 6

#Distortions
normalImages = "off"
motionAndEddyImages = "on"

#Noise
noiseLevel = [0,0.0081, 0.0165] #0.014, 0.023] #noise sigma

#Interleaving
interleaveFactor = 1;
#################################################################################################################

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
	if motionAndEddyImages == "on":
		simDirClusterDirectionMotionAndEddy = simDirCluster+"/DirectionMotionAndEddy"+str(direction)

		signal = readSignal(simDirClusterDirectionMotionAndEddy+'/signal')
		signalUninterleaved = unInterleaveSignal(signal,55,interleaveFactor)
		writeSignal(simDirClusterDirectionMotionAndEddy+'/signal',signalUninterleaved)

	if normalImages == "on":
		simDirClusterDirection = simDirCluster+"/Direction"+str(direction)


	#Generate noise
	for sigma in noiseLevel:
		if normalImages == "on":
			call(["systemnoise","-s",str(sigma),"-i",simDirClusterDirection+"/signal","-o",simDirClusterDirection+"/signalNoise"])
			call(["signal2image","-i",simDirClusterDirection+"/signalNoise","-p",simDirClusterDirection+"/pulse","-o",simDirClusterDirection+"/imageNoise","-a"])


		if motionAndEddyImages == "on":
			call(["systemnoise","-s",str(sigma),"-i",simDirClusterDirectionMotionAndEddy+"/signal","-o",simDirClusterDirectionMotionAndEddy+"/signalNoise"])
			call(["signal2image","-i",simDirClusterDirectionMotionAndEddy+"/signalNoise","-p",simDirClusterDirectionMotionAndEddy+"/pulse","-o",simDirClusterDirectionMotionAndEddy+"/imageNoise","-a"])

		#Save
		if motionAndEddyImages == "on":
			saveNoiseyImage(simDirClusterDirectionMotionAndEddy,resultsDir,"diff+eddy+motion_sigma{}_image{}.nii.gz".format(sigma,direction))
			convertImageToFloat(resultsDir,"diff+eddy+motion_sigma{}_image{}.nii.gz".format(sigma,direction))
		if normalImages == "on":
			saveNoiseyImage(simDirClusterDirection,resultsDir,"diff_sigma{}_image{}.nii.gz".format(sigma,direction))
			convertImageToFloat(resultsDir,"diff_sigma{}_image{}.nii.gz".format(sigma,direction))

#Merge
if motionAndEddyImages == "on":
	for sigma in noiseLevel:
		callMergeNoise = "fslmerge -a " + resultsDir + "/diff+eddy+motion_sigma{} ".format(sigma)
		callDelNoise = "rm "
		for i in range(numImages):
			callMergeNoise += resultsDir + "/diff+eddy+motion_sigma{}_image{}.nii.gz ".format(sigma,i)
			callDelNoise += resultsDir + "/diff+eddy+motion_sigma{}_image{}.nii.gz ".format(sigma,i)
		os.system(callMergeNoise)
		os.system(callDelNoise)


if normalImages == "on":
	pass
