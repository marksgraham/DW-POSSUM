#!/usr/bin/env python
#Code to generate a full shell of diffusion-weighted, eddy distorted images using FSL's possum, along with data that can be used to
#establish a ground truth.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import call
from dipy.io import read_bvals_bvecs
import dipy.reconst.shm as shm
import nibabel as nib
import possumLib as pl
import numpy as np
import scipy.io
import argparse
import shutil

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
parser.add_argument("--motion_dir",help='Path to directory describing subject motion.')
parser.add_argument("--interleave_factor",help="Interleave factor for slice-acquistion. Default=1. ",type=int,default=1)
parser.add_argument("--brain",help='Path to POSSUM input object.')
parser.add_argument("--brain_diffusion",help='Path to directory containing spherical harmonic coefficients for input object.')
parser.add_argument("--generate_artefact_free",help='Generate datasets without eddy-current and motion artefacts. Default=True.', type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("--generate_distorted",help='Generate datasets with eddy-current and motion artefacts. Default=False', type=str2bool, nargs='?',const=True,default=False)

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

if args.motion_dir == None:
		motionDir='None'
else:
	motionDir = args.motion_dir

if args.brain== None:
		segName = 'Files/Segmentations/HCP_seg_clipped.nii.gz'
else:
	segName  = args.brain

if args.brain_diffusion== None:
	sharm_dir = 'Files/SphericalHarmonics'
else:
	sharm_dir  = args.brain_diffusion
interleaveFactor = args.interleave_factor
normalImages = args.generate_artefact_free
motionAndEddyImages = args.generate_distorted

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

#Load in segmentations
print('Loading segmentation...')
segmentedBrain, segmentedBrainData = pl.loadSegData(segName)
print('Finished loading segmentation.')

#Load in spherical harmonic coefficients
order = 8;

#Check for problems first
if any(bval > 2500 for bval in bvals):
	raise NotImplementedError('bvals > 2000 currently not supported')

if any(bval > 100 and bval < 1500 for bval in bvals):
	print('Loading b=1000 spherical harmonics...')
	coefficientsNiib1000 = nib.load(os.path.join(sharm_dir,'coefficientsUpsampledb1000n'+str(order)+'.nii.gz'))
	coefficientsb1000 = coefficientsNiib1000.get_data()	
	print('Finished loading b=1000 spherical harmonics.')
if any(bval > 1500 and bval < 2500 for bval in bvals):
	print('Loading b=2000 spherical harmonics...')
	coefficientsNiib2000 = nib.load(os.path.join(sharm_dir,'coefficientsUpsampledb2000n'+str(order)+'.nii.gz'))
	coefficientsb2000 = coefficientsNiib2000.get_data()
	print('Finished loading b=2000 spherical harmonics.')




bvals, bvecs = read_bvals_bvecs(
	bvalDir, 
	bvecDir)

print('Output directory: ' + outputDir)


#Make directory for cluster files
pl.makeFolder(outputDir)
pl.makeFolder(outputDir+"/Results")
pl.makeFolder(outputDir+"/Distortions")
pl.makeFolder(outputDir+"/Distortions/Motion")
pl.makeFolder(outputDir+"/Distortions/Eddy")
if motionAndEddyImages == "on":
	shutil.copytree(motionDir, outputDir+"/Distortions/Motion")
	shutil.copy(simDir+"/pulse",outputDir+"/Distortions/Eddy")


#Move ref brain for registering
shutil.copy(simDir+"/brainref.nii.gz", outputDir)


for index, bvec in enumerate(bvecs[0:numImages]):
	#This workaround lets you carry on if generating is interrupted
	if index < 0:
		pass
	else:
		#Make directory for each setting
		outputDirDirection = outputDir+"/Direction"+str(index)
		pl.makeFolder(outputDirDirection)
		#Copy needed files to folder
		shutil.copy(simDir+"/MRpar",outputDirDirection)
		shutil.copy(simDir+"/motion",outputDirDirection)
		shutil.copy(simDir+"/slcprof",outputDirDirection)
		shutil.copy(simDir+"/pulse.info",outputDirDirection)
		shutil.copy(simDir+"/pulse.readme",outputDirDirection)
		shutil.copy(simDir+"/pulse.posx",outputDirDirection)
		shutil.copy(simDir+"/pulse.posy",outputDirDirection)
		shutil.copy(simDir+"/pulse.posz",outputDirDirection)
		shutil.copy(simDir+"/pulse",outputDirDirection)
		

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
			print('Attenuating volume ' + str(index))
			if bvals[index] < 1500:
				attenuatedBrainData = pl.attenuateImageSphericalHarmonics (segmentedBrainData, B, coefficientsb1000, bvals[index], 1000)
			elif  bvals[index] > 1500 and bvals[index] < 2500:
				attenuatedBrainData = pl.attenuateImageSphericalHarmonics (segmentedBrainData, B, coefficientsb2000, bvals[index], 2000)		




		attenuatedBrainNii = nib.Nifti1Image(attenuatedBrainData, segmentedBrain.get_affine(),segmentedBrain.get_header())

		attenuatedBrainNii.to_filename(os.path.join(codeDir,'attenuatedBrainPy.nii.gz'))

		shutil.move(codeDir + "/attenuatedBrainPy.nii.gz", outputDirDirection+ "/brain.nii.gz")

		#Register to reference brain to get sizes right
		print('Registering volume ' + str(index))
		call(["flirt","-in",outputDirDirection+ "/brain.nii.gz","-ref",outputDir+ "/brainref.nii.gz","-applyxfm","-out",outputDirDirection+ "/brain.nii.gz"])

		#Apply motion to brain here
		if motionAndEddyImages == True:
			outputDirDirectionMotionAndEddy = outputDir+"/DirectionMotionAndEddy"+str(index)
			shutil.copytree(outputDirDirection,outputDirDirectionMotionAndEddy)
			if motionDir is not "None":
				shutil.copy(motionDir + "/motion" + str(index) + '.txt', outputDirDirectionMotionAndEddy+ "/motion")

			#Read in pulse
			if index == 0:
				pulse=pl.read_pulse(outputDirDirection+"/pulse")
				pulseinfo = np.loadtxt(outputDirDirection+'/pulse.info')

			#Make distorted eddy pulse
			if eddyGradients=='flat':
				pl.generateEddyPulseFromBvecFlat(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)
			else:
				new_pulse = pl.addEddyAccordingToBvec(pulse,pulseinfo,basicSettings[0],basicSettings[1],basicSettings[2],basicSettings[3],ep, tau,bvals[index], bvec)

			#Interleave 
			if (interleaveFactor != 1):
				new_pulse = pl.interleavePulse(new_pulse,int(pulseinfo[12]),interleaveFactor)	

			#Save to correct location
			pl.write_pulse(outputDirDirectionMotionAndEddy+"/pulse",new_pulse)

			#Save copy of distorted pulse
			shutil.copy(outputDirDirectionMotionAndEddy+"/pulse", outputDir+"/Distortions/Eddy/pulseEddy"+str(index))


		if normalImages == False:
			shutil.rmtree(outputDirDirection)



