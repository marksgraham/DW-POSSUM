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
import argparse

#Parse arguments
parser = argparse.ArgumentParser(description="Setup all the files required to run the simulations.")

parser.add_argument("possum_dir",help="Path to the possum simulation directory.")
parser.add_argument("output_dir",help="Path to output directory.")
parser.add_argument("bvals",help="Path to bval file.")
parser.add_argument("bvecs",help="Path to bvec file.")
parser.add_argument("--num_images",help='Number of volumes. Defaults to number of entries in bval file.',type=int)
parser.add_argument("--motion_dir",help='Path to directory describing subject motion.')
parser.add_argument("--generate_artefact_free",help='Generate datasets without eddy-current and motion artefacts. Default=True.',action="store_true",default=1)
parser.add_argument("--generate_distorted",help='Generate datasets with eddy-current and motion artefacts. Default=False',action="store_true",default=0)

args=parser.parse_args()

#Check arguments and setup paths
simDir = os.path.abspath(args.possum_dir)
outputDirList = [args.output_dir]
bvalDirList = [args.bvals]
bvecDirList = [args.bvecs]
#Check bval entries
bvals, bvecs = read_bvals_bvecs(
		args.bvals, 
		args.bvecs)

#Check num_images
num_bvals = len(bvals)
if args.num_images:
	numImagesList= [args.num_images]
	if numImagesList[0] > num_bvals:
		print("Warning: num_images cannot be greater than the number of entries in the bval file. Setting to this maximum.")
		numImagesList[0] = num_bvals
else:
	numImagesList= [num_bvals]

if args.motion_dir == 0:
		motionDir=['None']
else:
	motionDir = [args.motion_dir]


normalImages = args.generate_artefact_free
motionAndEddyImages = args.generate_distorted



#Choose segmentation
#segName = 'HCP_seg_downsampled_clipped.nii.gz' 
segName = 'HCP_seg_clipped.nii.gz'


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

#Load in segmentations
segmentedBrain, segmentedBrainData = pl.loadSegData('Files/Segmentations',segName)


#Load in spherical harmonic coefficients
order = 8;

if any(bval > 100 and bval < 1500 for bval in bvals):
	print('Loading b=1000 spherical harmonics')
	coefficientsNiib1000 = nib.load(os.path.join('Files/SphericalHarmonics/coefficientsUpsampledb1000n'+str(order)+'.nii.gz'))
	coefficientsb1000 = coefficientsNiib1000.get_data()	
if any(bval > 1500 and bval < 2500 for bval in bvals):
	print('Loading b=2000 spherical harmonics')
	coefficientsNiib2000 = nib.load(os.path.join(
	'Files/SphericalHarmonics/coefficientsUpsampledb2000n'+str(order)+'.nii.gz'))
	coefficientsb2000 = coefficientsNiib2000.get_data()
if any(bval > 2500 for bval in bvals):
	raise NotImplementedError('bvals > 2000 currently not supported')

for dirNum, outputDir in enumerate(outputDirList):

	bvals, bvecs = read_bvals_bvecs(
		bvalDirList[dirNum], 
		bvecDirList[dirNum])
	
	print 'Output directory: ' + outputDir


	#Make directory for cluster files
	simDirCluster=outputDir
	call(["mkdir",simDirCluster])
	call(["mkdir",simDirCluster+"/Results"])
	call(["mkdir",simDirCluster+"/Distortions"])
	call(["mkdir",simDirCluster+"/Distortions/Motion"])
	call(["mkdir",simDirCluster+"/Distortions/Eddy"])
	pl.initialise(simDir,codeDir)
	if motionAndEddyImages == "on":
		call(["cp", "-r", motionDir[dirNum], simDirCluster+"/Distortions/Motion"])
		call(["cp",simDir+"/pulse",simDirCluster+"/Distortions/Eddy"])


	#Move ref brain for registering
	call(["cp",simDir+"/brainref.nii.gz", simDirCluster])


	for index, bvec in enumerate(bvecs[0:numImagesList[dirNum]]):
		#This workaround lets you carry on if generating is interrupted
		if index < 0:
			pass
		else:
			#Make directory for each setting
			simDirClusterDirection = simDirCluster+"/Direction"+str(index)
			call(["mkdir", simDirClusterDirection])
			#Copy needed files to folder
			call(["cp",simDir+"/MRpar",simDirClusterDirection])
			call(["cp",simDir+"/motion",simDirClusterDirection])
			call(["cp",simDir+"/slcprof",simDirClusterDirection])
			call(["cp",simDir+"/pulse.info",simDirClusterDirection])
			call(["cp",simDir+"/pulse.readme",simDirClusterDirection])
			call(["cp",simDir+"/pulse.posx",simDirClusterDirection])
			call(["cp",simDir+"/pulse.posy",simDirClusterDirection])
			call(["cp",simDir+"/pulse.posz",simDirClusterDirection])
			call(["cp",simDir+"/pulse",simDirClusterDirection])
			

			#Get attenuated segmentations
			#First rotate bvec
			#bvecRotated = pl.rotateBvecs(bvecs[index], motionParams[index,4:]);
			#Workaround: don't rotate bvec
			bvecRotated = bvecs[index]
			#print bvecs[index]
			#print bvecRotated

			if bvals[index] < 100:
				attenuatedBrainData = segmentedBrainData
			else:
				#Convert bvecs to angles
				x = bvecRotated[0]
				y = bvecRotated[1]
				z = bvecRotated[2]
				r, theta, phi = shm.cart2sphere(x, y, z)

				#Make design matrix
				B, m, n = shm.real_sym_sh_basis(order, theta, phi)
				
				#Get attenuated data
				if bvals[index] < 1500:
					attenuatedBrainData = pl.attenuateImageSphericalHarmonics (segmentedBrainData, B, coefficientsb1000, bvals[index], 1000)
				elif  bvals[index] > 1500 and bvals[index] < 2500:
					attenuatedBrainData = pl.attenuateImageSphericalHarmonics (segmentedBrainData, B, coefficientsb2000, bvals[index], 2000)		




			attenuatedBrainNii = nib.Nifti1Image(attenuatedBrainData, segmentedBrain.get_affine(),segmentedBrain.get_header())

			attenuatedBrainNii.to_filename(os.path.join(codeDir,'attenuatedBrainPy.nii.gz'))

			call(["mv",codeDir + "/attenuatedBrainPy.nii.gz", simDirClusterDirection+ "/brain.nii.gz"])

			#Register to reference brain to get sizes right
			call(["flirt","-in",simDirClusterDirection+ "/brain.nii.gz","-ref",simDirCluster+ "/brainref.nii.gz","-applyxfm","-out",simDirClusterDirection+ "/brain.nii.gz"])

			#Apply motion to brain here
			if motionAndEddyImages == True:
				simDirClusterDirectionMotionAndEddy = simDirCluster+"/DirectionMotionAndEddy"+str(index)
				call(["cp","-r",simDirClusterDirection,simDirClusterDirectionMotionAndEddy])
				if motionDir is not "None":
					call(["cp", motionDir[dirNum] + "/motion" + str(index) + '.txt', simDirClusterDirectionMotionAndEddy+ "/motion"  ])

				
				#Make distorted eddy pulse
				if eddyGradients=='flat':
					pl.generateEddyPulseFromBvecFlat(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)
				else:
					#pl.generateEddyPulseFromBvec(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)
					pl.addEddyAccordingToBvec(basicSettings[0],basicSettings[1],basicSettings[2],basicSettings[3],ep, tau,bvals[index], bvec)

				#Move eddy distorted pulse to simdir
				call(["mv",codeDir + "/pulse_new", simDirCluster+"/DirectionMotionAndEddy"+str(index)+"/pulse"])
				call(["cp", simDirCluster+"/DirectionMotionAndEddy"+str(index)+"/pulse", simDirCluster+"/Distortions/Eddy/pulseEddy"+str(index)])



			if normalImages == False:
				call(["rm","-rf", simDirClusterDirection])


	#Tidy up:
	pl.tidyUp(simDir,codeDir)

