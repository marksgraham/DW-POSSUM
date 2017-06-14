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

parser.add_argument("possum_dir")
parser.add_argument("output_dir")
parser.add_argument("bvals")
parser.add_argument("bvecs")

args=parser.parse_args()

simDir = os.path.abspath(possum_dir)
outputDirList = [output_dir];

print(args.possum_dir)

################Check the following are correct before running#################
# #Set relevant simulation directories:
# simDir = os.path.abspath('Files/POSSUMdirectories/possumSimdirOneSlice/')

# #set output directory
# outputDirList = ['Test/'];

# #Load in bvecs, bvals
# bvalDirList = ['Files/Bvalsbvecs/bvalsfmrib']
# bvecDirList = ['Files/Bvalsbvecs/bvecsfmrib']

# #Choose number of images to generate (must be <= length of bval file)
# numImagesList=[3];

# #Choose motion directory
# motionDir = ['None']

# #Choose whether to keep artefact-free images
# normalImages = "on";

# #Choose whether to generate distorted images
# motionAndEddyImages = "off";

###############################################################################
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
coefficientsNiib1000 = nib.load(os.path.join('Files/SphericalHarmonics/coefficientsUpsampledb1000n'+str(order)+'.nii.gz'))
coefficientsb1000 = coefficientsNiib1000.get_data()	

coefficientsNiib2000 = nib.load(os.path.join(
	'Files/SphericalHarmonics/coefficientsUpsampledb2000n'+str(order)+'.nii.gz'))
coefficientsb2000 = coefficientsNiib2000.get_data()

for dirNum, outputDir in enumerate(outputDirList):

	bvals, bvecs = read_bvals_bvecs(
		bvalDirList[dirNum], 
		bvecDirList[dirNum])
	print bvals
	print bvecs
	print outputDir


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
				else:
					print 'need to cater for higher bvals!'			




			attenuatedBrainNii = nib.Nifti1Image(attenuatedBrainData, segmentedBrain.get_affine(),segmentedBrain.get_header())

			attenuatedBrainNii.to_filename(os.path.join(codeDir,'attenuatedBrainPy.nii.gz'))

			call(["mv",codeDir + "/attenuatedBrainPy.nii.gz", simDirClusterDirection+ "/brain.nii.gz"])

			#Register to reference brain to get sizes right
			call(["flirt","-in",simDirClusterDirection+ "/brain.nii.gz","-ref",simDirCluster+ "/brainref.nii.gz","-applyxfm","-out",simDirClusterDirection+ "/brain.nii.gz"])

			#Apply motion to brain here
			if motionAndEddyImages == "on":
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



			if normalImages == "off":
				call(["rm","-rf", simDirClusterDirection])


	#Tidy up:
	pl.tidyUp(simDir,codeDir)

