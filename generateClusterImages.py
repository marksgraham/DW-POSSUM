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
#pl = reload(pl)

################Check the following are correct before running#################
#Set relevant simulation directories:
simDir = os.path.abspath('Files/possumSimdir')

#set output directory
outputDirList = ['../Test/simdirFulltest_code2/'];

#Load in bvecs, bvals
bvalDirList = ['test_code.bval']
bvecDirList = ['test_code.bvec']

#Choose number of images to generate (must be <= length of bval file)
numImagesList=[3];

#Choose motion file
motionFilesList = ['test_code.mat']

#Choose whether to keep artefact-free images
normalImages = "off";

#Choose whether to generate distorted images
motionAndEddyImages = "on";

###############################################################################
#Choose segmentation
#segName = 'HCP_seg_downsampled_clipped.nii.gz' 
segName = 'HCP_seg_clipped.nii.gz'


#Set eddy current distortion parameters
ep = 0.006 #default is 0.006
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
		'../Code/SimulateImages/bvalsbvecs/'+bvalDirList[dirNum], 
		'../Code/SimulateImages/bvalsbvecs/'+bvecDirList[dirNum])
	motionFile = '../Code/SimulateImages/motion/'+motionFilesList[dirNum]
	print bvals
	print bvecs
	print outputDir
	print motionFile

	#Load motion data
	if motionAndEddyImages == "on":
		motionParams = scipy.io.loadmat(motionFile)
		motionParams = motionParams["motionFile"]
		motionArray =np.zeros([2,7])
		print 'Motion:', motionArray

	#Make directory for cluster files
	simDirCluster=outputDir
	call(["mkdir",simDirCluster])
	call(["mkdir",simDirCluster+"/Results"])
	call(["mkdir",simDirCluster+"/ResultsNoise"])
	call(["mkdir",simDirCluster+"/Distortions"])
	call(["mkdir",simDirCluster+"/Distortions/Motion"])
	call(["mkdir",simDirCluster+"/Distortions/Eddy"])
	pl.initialise(simDir,codeDir)
	if motionAndEddyImages == "on":
		call(["cp", motionFile, simDirCluster+"/Distortions/Motion/motion.mat"])
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

			#Apply motion to brain here
			if motionAndEddyImages == "on":
				#Make FLIRT matrix
				pl.makeRotMatrix(motionParams[index,:],simDirClusterDirection)
				#Use FLIRT on the seg
				call(["flirt","-in",simDirClusterDirection+ "/brain.nii.gz","-ref", simDirClusterDirection+ "/brain.nii.gz","-out",simDirClusterDirection+ "/brainMotion.nii.gz", "-applyxfm","-init", "trans.mat"])
				#Register to ref
				call(["flirt","-in",simDirClusterDirection+ "/brainMotion.nii.gz","-ref",simDirCluster+ "/brainref.nii.gz","-applyxfm","-out",simDirClusterDirection+ "/brainMotion.nii.gz"])

				#Remove FLIRT matrix
				call(["rm", "trans.mat"])


				#Register to reference brain to get sizes right
				call(["flirt","-in",simDirClusterDirection+ "/brain.nii.gz","-ref",simDirCluster+ "/brainref.nii.gz","-applyxfm","-out",simDirClusterDirection+ "/brain.nii.gz"])


				call(["cp","-r",simDirClusterDirection,simDirCluster+"/DirectionMotionAndEddy"+str(index)])
			
				#Make distorted eddy pulse
				if eddyGradients=='flat':
					pl.generateEddyPulseFromBvecFlat(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)
				else:
					pl.generateEddyPulseFromBvec(simDir,codeDir,matlabDir,basicSettings,ep, tau,bvals[index], bvec)

				#Move eddy distorted pulse to simdir
				call(["mv",codeDir + "/pulse_new", simDirCluster+"/DirectionMotionAndEddy"+str(index)+"/pulse"])

				call(["cp", simDirCluster+"/Direction"+str(index)+"/brainMotion.nii.gz", simDirCluster+"/DirectionMotionAndEddy"+str(index)+"/brain.nii.gz"])
				call(["rm",simDirClusterDirection+ "/brainMotion.nii.gz"])
				call(["rm",simDirCluster+"/DirectionMotionAndEddy"+str(index)+ "/brainMotion.nii.gz"])
				call(["cp", simDirCluster+"/DirectionMotionAndEddy"+str(index)+"/pulse", simDirCluster+"/Distortions/Eddy/pulseEddy"+str(index)])

	

		if normalImages == "off":
			call(["rm","-rf", simDirClusterDirection])


	#Tidy up:
	pl.tidyUp(simDir,codeDir)
