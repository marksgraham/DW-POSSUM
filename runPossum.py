#!/usr/bin/env python

#Code to generate a full shell of diffusion-weighted, eddy distorted images using FSL's possum, along with data that can be used to
#establish a ground truth.

import os
from subprocess import call
import sys
import os.path
import argparse
#import possumLib as pl
#pl=reload(pl)

def saveImage(simDir,saveImageDir,fileName):
    call(["mv", simDir + "/image_abs.nii.gz", os.path.join(saveImageDir,fileName)])

def saveNoiseyImage(simDir,saveImageDir,fileName):
    call(["mv", simDir + "/imageNoise_abs.nii.gz", os.path.join(saveImageDir,fileName)])


parser = argparse.ArgumentParser(description="Run the simulations.")

parser.add_argument("simulation_dir",help="Path to the simulation directory (output_dir of generateFileStructure.py)")
parser.add_argument("num_images",help='Number of volumes.',type=int)
parser.add_argument("--simulate_artefact_free",help='Run simulation on datasets without eddy-current and motion artefacts. Default=True.',action="store_true",default=1)
parser.add_argument("--simulate_distorted",help='Run simulation datasets with eddy-current and motion artefacts. Default=False',action="store_true",default=0)
parser.add_argument("--num_processors",help='Number of processors to split each simulation over. Default=1',type=int,default=1)


args=parser.parse_args()

simDirCluster = os.path.abspath(args.simulation_dir)
numImages = args.num_images
normalImages = args.simulate_artefact_free
motionAndEddyImages = args.simulate_distorted
processors = args.num_processors 


resultsDir = simDirCluster+"/Results"
resultsNoiseDir = simDirCluster+"/ResultsNoise"

for direction in range(numImages):
	if normalImages == True:
		simDirClusterDirection = simDirCluster+"/Direction"+str(direction)
	if motionAndEddyImages == True:
		simDirClusterDirectionMotionAndEddy = simDirCluster+"/DirectionMotionAndEddy"+str(direction)
	
	#Run possum
	if normalImages == True:
		if os.path.isfile(simDirClusterDirection + '/image_abs.nii.gz') == False:
			call(["possumX", simDirClusterDirection,"-n",str(processors),"-t","20"])
	if motionAndEddyImages == True:
		if os.path.isfile(simDirClusterDirectionMotionAndEddy + '/image_abs.nii.gz') == False:
			call(["possumX", simDirClusterDirectionMotionAndEddy,"-n",str(processors),"-t","20"])


