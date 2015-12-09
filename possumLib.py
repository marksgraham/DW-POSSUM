from subprocess import call
import numpy as np
import os
import nibabel as nib
import pdb
import dipy.core.geometry as geom
import math

#Function copies some files needed by matlab from the simulation dir to the code dir, and saves backups of 
#key files from the simulation dir
def initialise(simDir, codeDir):
  call(["cp", simDir + "/pulse", codeDir])
  call(["cp", simDir + "/pulse.info", codeDir])
  call(["cp",simDir + "/pulse", simDir + "/pulseOld"])
  call(["cp",simDir + "/brain.nii.gz", simDir + "/brainOld.nii.gz"])

#Function tidies up everything done by initialise after code has run
def tidyUp(simDir,codeDir):
  call(["rm", codeDir + "/pulse"])
  call(["rm", codeDir + "/pulse.info"])
  call(["mv",simDir + "/pulseOld", simDir + "/pulse"])
  call(["mv",simDir + "/brainOld.nii.gz", simDir + "/brain.nii.gz"])


# Function takes as input simDir, codeDir, matlabDir and set of eddy pulse features - saves -new_pulse to codeDir
def generateEddyPulse(simDir,codeDir,matlabDir,bS, eS):
    str="addEddy({}, {}, {}, {}, {}, {}, {}, {}, {}, {});exit".format(bS[0],bS[1],bS[2],bS[3],eS[0],eS[1],eS[2],eS[3],eS[4],eS[5])
    call([matlabDir,  "-nodisplay", "-nojvm", "-nosplash","-r",str])
    
# Function takes as input sim dir, codedir and set of eddy pulse features including bvec, so the eddy currents 
# generated are a function of the applied bvec- saves -new_pulse to code dir
def generateEddyPulseFromBvec(simDir,codeDir,matlabDir,bS, ep, tau, bval,bvec):
  from subprocess import call
  str="addEddyAccordingToBvec({}, {}, {}, {}, {}, {}, {}, {}, {}, {});exit".format(bS[0],bS[1],bS[2],bS[3],ep, tau, bval, bvec[0], bvec[1], bvec[2])
  #print str
  call([matlabDir,  "-nodisplay", "-nojvm", "-nosplash","-r",str])

def generateEddyPulseFromBvecFlat(simDir,codeDir,matlabDir,bS, ep, tau, bval,bvec):
  from subprocess import call
  str="addEddyAccordingToBvecFlat({}, {}, {}, {}, {}, {}, {}, {}, {}, {});exit".format(bS[0],bS[1],bS[2],bS[3],ep, tau, bval, bvec[0], bvec[1], bvec[2])
  #print str
  call([matlabDir,  "-nodisplay", "-nojvm", "-nosplash","-r",str])


# Function that takes a segmented image, tensor image and a bval, bvec - attenuates the segmented image and returns
def attenuateImage(image,tensor,bval, bvec):
    
    #Rewrite bvectors into convenient form for multiplication with tensor
    bvecLong = np.array([0.0] * 6)
    bvecLong[0] = bvec[0] ** 2
    bvecLong[1] = bvec[0] * bvec[1] * 2
    bvecLong[2] = bvec[1] ** 2
    bvecLong[3] = bvec[0] * bvec[2] * 2
    bvecLong[4] = bvec[1] * bvec[2] * 2
    bvecLong[5] = bvec[2] ** 2
    
    #pdb.set_trace()
    attenuationMap = tensor * bvecLong
    attenuationMap=np.sum(attenuationMap, axis = 3)
    attenuationMap=np.exp(-(bval* (10 ** -6) * attenuationMap * (10 **6)))
   # img = attenuationMap[:,:,35]
   # implot=plt.imshow(img.squeeze())
   # plt.colorbar()
    #plt.show()
    attenuationMap = np.array([np.tile(attenuationMap, (1,1)) for i in xrange(3)]) 
    attenuationMap = np.transpose(attenuationMap,[1,2,3,0])
    attenuatedImage =  attenuationMap * image
    return attenuatedImage, attenuationMap


def attenuateImageSphericalHarmonics (image, B, coefficients,bval, bvalStandard):
  dataSize = coefficients.shape
  attenuationMap = np.zeros((dataSize[0],dataSize[1],dataSize[2]))
  for i in range(0,dataSize[0]):
    for j in range(0,dataSize[1]):
      for k in range(0,dataSize[2]):
        coeff = coefficients[i,j,k,:] 
        attenuationMap[i,j,k] = np.dot(B,coeff)

  attenuationMap = np.array([np.tile(attenuationMap, (1,1)) for i in xrange(3)]) 
  attenuationMap = np.transpose(attenuationMap,[1,2,3,0])
  #Correct for b-value not being exactly 1000/2000 using mono-exponential assumption:
  if abs(bval-bvalStandard) > 50:
    attenuationMap=np.log(attenuationMap) * (bval/bvalStandard)
    attenuationMap=np.exp(attenuationMap)
  
  attenuatedImage =  attenuationMap * image
  return attenuatedImage

def saveImage(simDir,saveImageDir,fileName):
  call(["mv", simDir + "/image_abs.nii.gz", os.path.join(saveImageDir,fileName)])

#Load in segmented brain used for possum
def loadSegData(templateDir,segmentedName):
  segmentedBrain = nib.load(os.path.join(templateDir,segmentedName))
  segmentedBrainData = segmentedBrain.get_data()
  return segmentedBrain, segmentedBrainData

#Load in tensor data
def loadTensorData(templateDir,templateName):
  tensorBrain = nib.load(os.path.join(templateDir,templateName))
  tensorBrainData = tensorBrain.get_data()
  return tensorBrainData

def makeRotMatrix(motionParams, simDirClusterDirection):
  #Make three rotation matrices
  call(["makerot", "-t", str(motionParams[4]), "-a", "1,0,0", "--cov="+simDirClusterDirection+ "/brain.nii.gz", "-o", "rotx.mat"])
  call(["makerot", "-t", str(motionParams[5]), "-a", "0,1,0", "--cov="+simDirClusterDirection+ "/brain.nii.gz", "-o", "roty.mat"])
  call(["makerot", "-t", str(motionParams[6]), "-a", "0,0,1", "--cov="+simDirClusterDirection+ "/brain.nii.gz", "-o", "rotz.mat"])
  #Concatenate
  call(["convert_xfm", "-omat", "rotxy.mat","-concat", "roty.mat", "rotx.mat"])
  call(["convert_xfm", "-omat", "rotxyz.mat","-concat", "rotz.mat", "rotxy.mat"])

  #Add translations
  rot = np.loadtxt('rotxyz.mat')
  rot[0,3] += motionParams[1]
  rot[1,3] += motionParams[2]
  rot[2,3] += motionParams[3]
  np.savetxt('trans.mat', rot )
  #Tidy up
  call(["rm","rotx.mat","roty.mat","rotz.mat","rotxy.mat","rotxyz.mat",])

def rotateBvecs(bvec, rotationAngles):
  #Rotates the bvecs before the attentuation is sampled, based on motion parameters, so that rotations adjust the diffusion contrast. Code rotates the brain around x, y then z sucessively, so this code rotates the bvec around z, y then x.
  rotMat = geom.euler_matrix(math.radians(rotationAngles[2]), math.radians(rotationAngles[1]), math.radians(rotationAngles[0]), 'szyx')

  
  bvecRot = np.dot(rotMat[0:3,0:3],bvec)
  return bvecRot
