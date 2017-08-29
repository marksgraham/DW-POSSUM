#!/usr/bin/env python
#Code to generate the input segmentions for a full shell of diffusion-weighted, eddy distorted images using FSL's possum.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Handle arguments (before slow imports so --help can be fast)
import argparse
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
parser.add_argument("--brain",help='Path to POSSUM input object.')
parser.add_argument("--brain_diffusion",help='Path to directory containing spherical harmonic coefficients for input object.')
args=parser.parse_args()

import os
import sys
from subprocess import call
from dipy.io import read_bvals_bvecs
import dipy.reconst.shm as shm
import nibabel as nib
import numpy as np
import scipy.io
import shutil

#Add root to pythonpath for lib import
dir_path = os.path.dirname(os.path.realpath(__file__))
package_path = os.path.abspath(os.path.join(dir_path,os.pardir))
sys.path.insert(0,package_path)
from Library import possumLib as pl

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


if args.brain== None:
		segName = '../Files/Segmentations/HCP_seg_clipped.nii.gz'
else:
	segName  = args.brain

if args.brain_diffusion== None:
	sharm_dir = '../Files/SphericalHarmonics'
else:
	sharm_dir  = args.brain_diffusion



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

#Move ref brain for registering
shutil.copy(simDir+"/brainref.nii.gz", outputDir)


for index, bvec in enumerate(bvecs[0:numImages]):
	#This workaround lets you carry on if generating is interrupted
	if index < 0:
		pass
	else:

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

		shutil.move(codeDir + "/attenuatedBrainPy.nii.gz", outputDir+ "/brain"+str(index)+".nii.gz")

		#Register to reference brain to get sizes right
		print('Registering volume ' + str(index))
		call(["flirt","-in",outputDir+ "/brain"+str(index)+".nii.gz","-ref",outputDir+ "/brainref.nii.gz","-applyxfm","-out",outputDir+ "/brain"+str(index)+".nii.gz"])

