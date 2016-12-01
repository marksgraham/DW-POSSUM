#Code to generate weighted, high resolution segmentations
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
simDir = os.path.abspath('Files/possumSimdirSliceRes4')

#set output directory
outputDirList = ['/Users/markgraham/Dropbox/UCL/Projects/EddyCurrents/Simulation/MotionWithSus/WeightedSegsb1000/'];

#Load in bvecs, bvals
bvalDirList = ['simulationb1000.bval']
bvecDirList = ['simulation.bvec']

#Choose number of images to generate (must be <= length of bval file)
numImagesList=[36];


#Choose whether to keep artefact-free images
normalImages = "on";


###############################################################################
#Choose segmentation
#segName = 'HCP_seg_downsampled_clipped.nii.gz' 
segName = 'HCP_seg_clipped.nii.gz'


#Set directories
codeDir =  os.path.abspath('.')
matlabDir = "/Applications/MATLAB_R2014b.app/bin/matlab"

#Load in segmentations
segmentedBrain, segmentedBrainData = pl.loadSegData('Files/Segmentations',segName)


#Load in spherical harmonic coefficients
order = 8;
coefficientsNiib1000 = nib.load(os.path.join('Files/SphericalHarmonics/coefficientsUpsampledb1000n'+str(order)+'.nii.gz'))
coefficientsb1000 = coefficientsNiib1000.get_data()	

coefficientsNiib2000 = nib.load(os.path.join(
	'Files/SphericalHarmonics/coefficientsUpsampledb2000n'+str(order)+'.nii.gz'))
coefficientsb2000 = coefficientsNiib2000.get_data()

for dirNum, outputDir in enumerate(outputDirList):

	bvals, bvecs = read_bvals_bvecs(
		'../Code/SimulateImages/bvalsbvecs/'+bvalDirList[dirNum], 
		'../Code/SimulateImages/bvalsbvecs/'+bvecDirList[dirNum])
	print bvals
	print bvecs
	print outputDir


	#Make directory for cluster files
	simDirCluster=outputDir
	call(["mkdir",simDirCluster])

	for index, bvec in enumerate(bvecs[0:numImagesList[dirNum]]):
		#This workaround lets you carry on if generating is interrupted
		if index < 0:
			pass
		else:
			bvec = bvecs[index]

			#Get attenuated segmentations
			if bvals[index] < 100:
				attenuatedBrainData = segmentedBrainData
			else:
				#Convert bvecs to angles
				x = bvec[0]
				y = bvec[1]
				z = bvec[2]
				r, theta, phi = shm.cart2sphere(x, y, z)

				#Make design matrix
				B, m, n = shm.real_sym_sh_basis(order, theta, phi)
				
				#Get attenuated data
				if bvals[index] < 1500:
					attenuatedBrainData = pl.attenuateImageSphericalHarmonics (segmentedBrainData, B, coefficientsb1000, bvals[index], 1000)
				elif  bvals[index] > 1500 and bvals[index] < 2500:
					attenuatedBrainData = pl.attenuateImageSphericalHarmonics (segmentedBrainData, B, coefficientsb2000, bvals[index], 2000)
				else:
					print 'need to cater for higher bvals!'			




			attenuatedBrainNii = nib.Nifti1Image(attenuatedBrainData, segmentedBrain.get_affine(),segmentedBrain.get_header())

			attenuatedBrainNii.to_filename(os.path.join(simDirCluster,'weightedSeg' + str(index) + '.nii.gz'))


			#Register to reference brain to get sizes right
			call(["flirt","-in",simDirCluster +'/weightedSeg' + str(index) + '.nii.gz',"-ref",simDir+ "/brainref.nii.gz","-applyxfm","-out",simDirCluster +'/weightedSeg' + str(index) + '.nii.gz'])



