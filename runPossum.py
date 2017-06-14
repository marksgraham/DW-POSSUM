#!/usr/bin/env python

#Code to generate a full shell of diffusion-weighted, eddy distorted images using FSL's possum, along with data that can be used to
#establish a ground truth.

import os
from subprocess import call
import sys
import os.path
#import possumLib as pl
#pl=reload(pl)

def saveImage(simDir,saveImageDir,fileName):
    call(["mv", simDir + "/image_abs.nii.gz", os.path.join(saveImageDir,fileName)])

def saveNoiseyImage(simDir,saveImageDir,fileName):
    call(["mv", simDir + "/imageNoise_abs.nii.gz", os.path.join(saveImageDir,fileName)])
################################### CHECK THIS PART ###########################################################
#Choose relevant directories:
simDirCluster = os.path.abspath('Test/')
numImages = 3
normalImages = "on"
eddyImages = "off" #on or off
motionImages = "off"
motionAndEddyImages = "off"
processors = 1 #split each image up this many times for parallelisation
###############################################################################################################

resultsDir = simDirCluster+"/Results"
resultsNoiseDir = simDirCluster+"/ResultsNoise"

for direction in range(numImages):
	if normalImages == "on":
		simDirClusterDirection = simDirCluster+"/Direction"+str(direction)
	if eddyImages == "on":
		simDirClusterDirectionEddy = simDirCluster+"/DirectionEddy"+str(direction)
	if motionImages == "on":
		simDirClusterDirectionMotion = simDirCluster+"/DirectionMotion"+str(direction)
	if motionAndEddyImages == "on":
		simDirClusterDirectionMotionAndEddy = simDirCluster+"/DirectionMotionAndEddy"+str(direction)
	
	#Run possum
	if normalImages == "on":
		if os.path.isfile(simDirClusterDirection + '/image_abs.nii.gz') == False:
			call(["possumX", simDirClusterDirection,"-n",str(processors),"-t","20"])
	if eddyImages == "on":
		call(["possumX", simDirClusterDirectionEddy,"-n",str(processors),"-t","20"])
	if motionImages == "on":
		call(["possumX", simDirClusterDirectionMotion,"-n",str(processors),"-t","20"])
	if motionAndEddyImages == "on":
		if os.path.isfile(simDirClusterDirectionMotionAndEddy + '/image_abs.nii.gz') == False:
			call(["possumX", simDirClusterDirectionMotionAndEddy,"-n",str(processors),"-t","20"])


