from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from subprocess import call
import numpy as np
import os
import nibabel as nib
import pdb
import dipy.core.geometry as geom


import math

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
    attenuationMap = np.array([np.tile(attenuationMap, (1,1)) for i in range(3)]) 
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

  attenuationMap = np.array([np.tile(attenuationMap, (1,1)) for i in range(3)]) 
  attenuationMap = np.transpose(attenuationMap,[1,2,3,0])

  #Get rid of any negative values
  attenuationMap[attenuationMap < 0] = 0
  #Correct for b-value not being exactly 1000/2000 using mono-exponential assumption:
  if abs(bval-bvalStandard) > 50:
    attenuationMap=np.log(attenuationMap) * (bval/bvalStandard)
    attenuationMap=np.exp(attenuationMap)
  
  attenuatedImage =  attenuationMap * image
  return attenuatedImage

def saveImage(simDir,saveImageDir,fileName):
  call(["mv", simDir + "/image_abs.nii.gz", os.path.join(saveImageDir,fileName)])

#Load in segmented brain used for possum
def loadSegData(seg_path):
  segmentedBrain = nib.load(seg_path)
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

def read_pulse(fname):
    #Implementation of the read_pulse.m shipped with FSL. This is stripped down and assumes little-endian storage.
    fid = open(fname,"rb")
    testval= np.fromfile(fid, dtype=np.uint32,count = 1)
    dummy=np.fromfile(fid, dtype=np.uint32,count = 1)
    nrows=np.fromfile(fid, dtype=np.uint32,count = 1)
    ncols=np.fromfile(fid, dtype=np.uint32,count = 1)
    nrows= int(nrows)
    ncols= int(ncols)

    time=np.fromfile(fid, dtype=np.float64,count = nrows)
    mvals = np.fromfile(fid, dtype=np.float32, count = nrows*(ncols-1))
    mvals = np.reshape(mvals,(nrows, ncols-1),order='F')
    m = np.zeros((nrows,ncols))
    m[:,0] = time
    m[:,1:] = mvals

    fid.close()
    return m

def write_pulse(fname,mat):
    time = mat[:,0]
    mvals = mat[:,1:]

    fidin = open(fname,"w")  
    magicnumber=42+1
    dummy=0
    [nrows,ncols]=mat.shape
    header = np.array([magicnumber,dummy,nrows,ncols])
    header.astype(np.uint32).tofile(fidin)
    time.astype(np.float64).tofile(fidin)
    mvals = np.reshape(mvals,[len(mvals)*7],order='F')
    mvals.astype(np.float32).tofile(fidin)
    fidin.close()

def addEddyAccordingToBvec(pulse,pulseinfo,tint,delta,Delta,Gdiff,ep,tau,bval,bvec):
    #Quick hack to ensure eddy currents scale with b-value
    Gdiff=Gdiff * bval / 2000

    #Extract time
    time=pulse[:,0].T
    numSlices=int(pulseinfo[12])
    TRslice=pulseinfo[3]
    RFtime=time[7]
    #Adjust RF time for spin-echo pulse sequences
    if pulseinfo[0]==3:
      RFtime=time[14]

    Eddyx=np.zeros((4,len(pulse)))
    Eddyy=np.zeros((4,len(pulse)))
    Eddyz=np.zeros((4,len(pulse)))

    for i in range(numSlices):
        t = np.zeros(4)
        t[0]=tint + TRslice * i
        t[1]=tint + delta + TRslice * i
        t[2]=tint + Delta + TRslice * i
        t[3]=tint + delta + Delta + TRslice * i
        RF=RFtime + TRslice * i
        #sliceEnd allows the eddy current to be cut off at the end of the slice acquistion, preventing the slice-wise buildup of ECs. This allows me to shorten TRslice to a realistic value in the simulations whilst not needing to model slice-by-slice buildup for now.
        sliceEnd = TRslice * ( i + 1)

        #Remember the signs need to change
        for j in range(4):
            logical = (time > t[j]) * (time > RF) * (time < sliceEnd)
            addx=(ep * Gdiff * bvec[0] * (np.exp(- (time - t[j]) / tau))) * logical
            addy=(ep * Gdiff * bvec[1] * (np.exp(- (time - t[j]) / tau))) * logical
            addz=(ep * Gdiff * bvec[2] * (np.exp(- (time - t[j]) / tau))) * logical
            addx[np.isnan(addx)]=0
            addy[np.isnan(addy)]=0
            addz[np.isnan(addz)]=0
            if j == 1 or j == 2:
                addx=addx * - 1
                addy=addy * - 1
                addz=addz * - 1
            Eddyx[j,:]=Eddyx[j,:] + addx
            Eddyy[j,:]=Eddyy[j,:] + addy
            Eddyz[j,:]=Eddyz[j,:] + addz
    new_pulse=pulse
    new_pulse[:,5]=pulse[:,5] + np.sum(Eddyx,0).T
    new_pulse[:,6]=pulse[:,6] + np.sum(Eddyy,0).T
    new_pulse[:,7]=pulse[:,7] + np.sum(Eddyz,0).T
    return new_pulse

def makeFolder(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def interleavePulse(pulse, numSlices, interleaveFactor):
  [nrows,ncols]=pulse.shape
  pulseInterleaved = np.zeros((nrows,ncols))
  pulseInterleaved[:,0] = pulse[:,0]
  counter = 0
  entriesPerSlice = int(nrows/numSlices)
  for i in range(interleaveFactor):
    for j in range(i,numSlices,interleaveFactor):
      startIndexOld = counter* entriesPerSlice
      endIndexOld = (counter + 1) * entriesPerSlice -1
      startIndex = j* entriesPerSlice
      endIndex = (j + 1) * entriesPerSlice -1
      pulseInterleaved[startIndexOld:endIndexOld,1:] = pulse[startIndex:endIndex,1:]
      counter = counter + 1
  return pulseInterleaved

def get_motion_level(directory,translation_threshold=2.5,rotation_threshold=2.5):
  '''Roughly classify the amount of motion per slice for calculating likelihood of signal dropouts - 1 = severe motion, 2 = moderate motion'''
  motion_path = os.path.join(directory,'motion')
  motion = np.loadtxt(motion_path)
  max_motion = np.max(motion[:,1:],axis=0)
  min_motion = np.min(motion[:,1:],axis=0)
  diff_motion = np.abs(max_motion-min_motion)
  diff_motion[:3] = diff_motion[:3]*1000
  diff_motion[3:] = np.rad2deg(diff_motion[3:])

  if np.any( diff_motion[:3] > translation_threshold):
      return 1    
  elif np.any(diff_motion[3:] > rotation_threshold):
      return 1
  elif np.any( diff_motion[:3] > 1):
      return 2    
  elif np.any(diff_motion[3:] > 1):
      return 2
  else:
      return 0

def add_signal_dropout(signal,motion_level,num_slices,num_voxels_per_slice):
  if motion_level ==0:
    if np.random.random() < 0: #Add dropout  volumes with no motion
      for i in range(num_slices):
        if np.random.random() < 0.4:
          dropout_factor = np.random.random()
          signal[:,num_voxels_per_slice*i:num_voxels_per_slice*(i+1)] = signal[:,num_voxels_per_slice*i:num_voxels_per_slice*(i+1)] * dropout_factor
  
  if motion_level == 1:
    if np.random.random() < 0.3: #Add dropout to 30% of volumes with severe motion
     for i in range(num_slices):
       if np.random.random() < 0.4:
         dropout_factor = np.random.random()
         signal[:,num_voxels_per_slice*i:num_voxels_per_slice*(i+1)] = signal[:,num_voxels_per_slice*i:num_voxels_per_slice*(i+1)] * dropout_factor

  elif motion_level == 2: #Don't add dropout to volumes with moderate motion atm
    for i in range(num_slices):
      if np.random.random() < 0.0: #Dont add to examples with mild motion
        dropout_factor = np.random.random()
        signal[:,num_voxels_per_slice*i:num_voxels_per_slice*(i+1)] = signal[:,num_voxels_per_slice*i:num_voxels_per_slice*(i+1)] * dropout_factor
  return signal



