#!/usr/bin/env python

#Generate POSSUM input object and spherical harmonic coefficients from an HCP dataset
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(description="Generate POSSUM input object and spherical harmonic coefficients from an HCP dataset")
parser.add_argument("t1",help="Path to T1 image")
parser.add_argument("t2",help="Path to T2 image")
parser.add_argument("dwi",help="Path to DWI directory")
parser.add_argument("out",help="Path to output directory")
parser.add_argument("--bvalues",help="B-values to generate spherical harmonics for. Default b=1000 only.",nargs="+",type=int)
parser.add_argument("--csf_factor",help="A factor < 1 provides extra signal attenuation in the CSF. Default 1. ",type=float)
args=parser.parse_args()

#Now imports
import os
from subprocess import call, Popen, PIPE
from dipy.io import read_bvals_bvecs
from dipy.external.fsl import write_bvals_bvecs
from Library import possumLib as pl
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
import dipy.reconst.shm as shm

#Assign args
t1=os.path.abspath(args.t1)
t2=os.path.abspath(args.t2)
dwi=os.path.abspath(args.dwi)
out=os.path.abspath(args.out)
if args.bvalues == None:
    bvalues = [1000]
else:
    bvalues = args.bvalues
if args.csf_factor == None:
    csf_factor = 1.0
else:
    csf_factor = args.csf_factor

print('\n')
print('T1 image: ' + t1)
print('T2 image: ' + t2)
print('DWI directory: ' + dwi)
print('Output directory: ' + out)

if not os.path.exists(out):
	os.makedirs(out)

#Use FSL's FAST to segment 
call(["fast","-S","2","-n","3","--verbose","-o",out+"/fast","-p",t1,t2])
#Combine into seg in order 1. GM 2. WM 3.CSF. Normally, fast outputs segmentations in order WM, CSF, GM but occasionally does CSF WM GM - need to check for this case.
tissue_mean = np.zeros((3))
for i in range(3):
    p=Popen(['fslstats',out+'/fast_pve_'+str(i)+'.nii.gz', '-m'],stdout=PIPE)
    output= p.communicate()
    tissue_mean[i] = float(output[0])
if tissue_mean[1] < tissue_mean[0]:
    call(["fslmerge","-a",out+'/HCP_seg.nii.gz',out+'/fast_prob_2',out+'/fast_prob_0',out+'/fast_prob_1'])
else:
    call(["fslmerge","-a",out+'/HCP_seg.nii.gz',out+'/fast_prob_2',out+'/fast_prob_1',out+'/fast_prob_0'])

#Extract diffusion shells
bvals, bvecs = read_bvals_bvecs(dwi+'/bvals', dwi+'/bvecs')
call(['fslsplit', dwi+'/data.nii.gz',out+'/vol'])


bvals1000Trimmed=[]
bvals2000Trimmed=[]
bvals3000Trimmed=[]

bvecs1000Trimmed=[]
bvecs2000Trimmed=[]
bvecs3000Trimmed=[]
pl.makeFolder(out+"/DiffusionReordered/b1000")
pl.makeFolder(out+"/DiffusionReordered/b2000")
pl.makeFolder(out+"/DiffusionReordered/b3000")

print(bvecs.shape)

for index, bval in enumerate(bvals):
	if bval < 100:
		bvals1000Trimmed.append(bval)
		bvecs1000Trimmed.append(bvecs[index,:])

		bvals2000Trimmed.append(bval)
		bvecs2000Trimmed.append(bvecs[index,:])		

		bvals3000Trimmed.append(bval)
		bvecs3000Trimmed.append(bvecs[index,:])

		call(["cp",out+"/vol{:0>4d}.nii.gz".format(index),out+"/DiffusionReordered/b1000/vol{:0>4d}.nii.gz".format(index)])

		call(["cp",out+"/vol{:0>4d}.nii.gz".format(index),out+"/DiffusionReordered/b2000/vol{:0>4d}.nii.gz".format(index)])

		call(["cp",out+"/vol{:0>4d}.nii.gz".format(index),out+"/DiffusionReordered/b3000/vol{:0>4d}.nii.gz".format(index)])

		call(["rm",out+"/vol{:0>4d}.nii.gz".format(index)])

	if bval > 900 and bval <1100:
		bvals1000Trimmed.append(bval)
		bvecs1000Trimmed.append(bvecs[index,:])
		call(["cp",out+"/vol{:0>4d}.nii.gz".format(index),out+"/DiffusionReordered/b1000/vol{:0>4d}.nii.gz".format(index)])
		call(["rm",out+"/vol{:0>4d}.nii.gz".format(index)])

	if bval > 1900 and bval <2100:
		bvals2000Trimmed.append(bval)
		bvecs2000Trimmed.append(bvecs[index,:])
		call(["cp",out+"/vol{:0>4d}.nii.gz".format(index),out+"/DiffusionReordered/b2000/vol{:0>4d}.nii.gz".format(index)])
		call(["rm",out+"/vol{:0>4d}.nii.gz".format(index)])

	if bval > 2900 and bval <3100:
		bvals3000Trimmed.append(bval)
		bvecs3000Trimmed.append(bvecs[index,:])
		call(["cp",out+"/vol{:0>4d}.nii.gz".format(index),out+"/DiffusionReordered/b3000/vol{:0>4d}.nii.gz".format(index)])
		call(["rm",out+"/vol{:0>4d}.nii.gz".format(index)])


#Save bval and bvec files
write_bvals_bvecs(bvals1000Trimmed,bvecs1000Trimmed,prefix=out+'/DiffusionReordered/b1000/')

write_bvals_bvecs(bvals2000Trimmed,bvecs2000Trimmed,prefix=out+'/DiffusionReordered/b2000/')
write_bvals_bvecs(bvals3000Trimmed,bvecs3000Trimmed,prefix=out+'/DiffusionReordered/b3000/')

#Merge and clean up
call(['fslmerge -a ' + out + '/DiffusionReordered/b1000/data.nii.gz ' + out + '/DiffusionReordered/b1000/vol0*'], shell=True)
call(['rm ' + out + '/DiffusionReordered/b1000/vol0*'], shell=True)

call(['fslmerge -a ' + out + '/DiffusionReordered/b2000/data.nii.gz ' + out + '/DiffusionReordered/b2000/vol0*'], shell=True)
call(['rm ' + out + '/DiffusionReordered/b2000/vol0*'], shell=True)

call(['fslmerge -a ' + out + '/DiffusionReordered/b3000/data.nii.gz ' + out + '/DiffusionReordered/b3000/vol0*'], shell=True)
call(['rm ' + out + '/DiffusionReordered/b3000/vol0*'], shell=True)

#b1000 data
#call(["dtifit","-k",out + "/DiffusionReordered/b1000/data","-o",out+"/DT","-m",dwi+"/nodif_brain_mask.nii.gz","-r",out + "/DiffusionReordered/b1000/bvecs","-b",out + "/DiffusionReordered/b1000/bvals","--verbose","--save_tensor"])

#b2000 data
#call(["dtifit","-k",out + "/DiffusionReordered/b2000/data","-o",out+"/DT","-m",dwi+"/nodif_brain_mask.nii.gz","-r",out + "/DiffusionReordered/b2000/bvecs","-b",out + "/DiffusionReordered/b2000/bvals","--verbose","--save_tensor"])

#All data
call(["dtifit","-k",dwi + "/data","-o",out+"/DT","-m",dwi+"/nodif_brain_mask.nii.gz","-r",dwi + "/bvecs","-b",dwi + "/bvals","--verbose","--save_tensor"])

#Upsample tensor to segmentation space
os.system('fslhd -x ' + out + '/HCP_seg.nii.gz > ' + out + '/hdr.txt')
call(['fslcreatehd', out + '/hdr.txt', out + '/brainref'])
for i in range(1,4):
	call(['flirt', '-in', out + '/DT_L' + str(i), '-ref', out + '/brainref', '-out', out + '/DT_L' + str(i) +'_upsampled', '-applyxfm'])

#Use DT data to fine-tune the segmentation
seg = nib.load(out + '/HCP_seg.nii.gz')
seg_data = seg.get_data()	
for i in range(1,4):
	eigenvalue = nib.load(out + '/DT_L' + str(i) +'_upsampled.nii.gz')
	eigenvalue_data = eigenvalue.get_data()
	#Set voxels with negative eigenvalues to 0
	for j in range(3):
		seg_data[:,:,:,j][eigenvalue_data < 0] = 0
		#Set voxels with very low diffusivities to 0
		if (i == 1):
			seg_data[:,:,:,j][eigenvalue_data < 0.0003] = 0

#Normalise seg
seg_sum = np.sum(seg_data,axis=3)
for j in range(3):
	with np.errstate(divide='ignore',invalid='ignore'):
		seg_data[:,:,:,j] = seg_data[:,:,:,j] / seg_sum
seg_data[np.isnan(seg_data)] =0
#Save
seg_clipped = nib.Nifti1Image(seg_data, seg.get_affine(),seg.get_header())
seg_clipped.to_filename(out + '/HCP_seg_clipped_new.nii.gz')

#Fit SH to every shell independently and upsample
#Choose order
order =8;

# Directories
for bvalue in bvalues:
    diffusionDir = out +'/DiffusionReordered/b' + \
        str(bvalue)

    saveDir = out + '/SphericalHarmonics'
    pl.makeFolder(saveDir)
    dwiDir = os.path.join(diffusionDir, 'data.nii.gz')
    bvalDir = os.path.join(diffusionDir, 'bvals')
    bvecDir = os.path.join(diffusionDir, 'bvecs')
    maskDir = os.path.join(dwi, 'nodif_brain_mask.nii.gz')

    # Load data
    img = nib.load(dwiDir)
    data = img.get_data()
    maskImg = nib.load(maskDir)
    maskData = maskImg.get_data()
    bvals, bvecs = read_bvals_bvecs(bvalDir, bvecDir)
    gtab = gradient_table(bvals, bvecs,b0_threshold=50)

    # Normalise data
    dataNormalised = shm.normalize_data(data, gtab.b0s_mask)
    # dataNormalisedNii = nib.Nifti1Image(dataNormalised, img.get_affine(),
    # img.get_header())
    # dataNormalisedNii.to_filename('dataNormalised.nii.gz')

    # Convert bvecs to angles
    where_dwis = 1 - gtab.b0s_mask
    x = gtab.gradients[where_dwis == True, 0]
    y = gtab.gradients[where_dwis == True, 1]
    z = gtab.gradients[where_dwis == True, 2]
    r, theta, phi = shm.cart2sphere(x, y, z)

    # Make design matrix
    B, m, n = shm.real_sym_sh_basis(order, theta[:, None], phi[:, None])
    Binverse = shm.pinv(B)

    # Make matrix to hold coefficients
    dataSize = data.shape
    coefficientArray = np.zeros(
        (dataSize[0], dataSize[1], dataSize[2], len(B[1, :])))

    # Get coefficients
    for i in range(0, dataSize[0]):
        for j in range(0, dataSize[1]):
            for k in range(0, dataSize[2]):
                if maskData[i, j, k] != 0:
                    dataColumn = dataNormalised[i, j, k, where_dwis == True]
                    coefficientArray[i, j, k] = np.dot(Binverse, dataColumn)

    # Save coefficients
    fName = 'coefficientsb' + str(bvalue) + 'n' + str(order) + '.nii.gz'
    coefficientNii = nib.Nifti1Image(coefficientArray, img.get_affine())
    coefficientNii.header.set_data_dtype('float32')
    coefficientNii.to_filename(os.path.join(saveDir, fName))

    #Free memory
    del data, maskData

    # #Predict back data
    # dataPredicted = np.zeros((dataSize[0],dataSize[1],dataSize[2],
    # sum(where_dwis)))
    # for i in range(0,dataSize[0]):
    # 	for j in range(0,dataSize[1]):
    # 		for k in range(0,dataSize[2]):
    # 			if maskData[i,j,k] != 0:
    # 				coeff = coefficientArray[i,j,k,:]
    # 				dataPredicted[i,j,k,:] = np.dot(B,coeff)

    # dataPredictedNii = nib.Nifti1Image(dataPredicted,img.get_affine())
    # dataPredictedNii.header.set_data_dtype('float32')
    # predFName = 'dataPredictedb' + str(bvalue)+'.nii.gz'
    # dataPredictedNii.to_filename(os.path.join(saveDir,predFName))

    # Upsample coefficients to segmentation space
    outName = 'coefficientsUpsampledb' + str(bvalue) + 'n' + str(order) + '.nii.gz'
    segRef = os.path.join(
        out, 'brainref.nii.gz')
    call(['flirt', '-in', os.path.join(saveDir, fName), '-ref',
          segRef, '-out', os.path.join(saveDir, outName), '-applyxfm'])

    #Post-process: alter CSF values
    #Load upsampled coeffs and seg
    coeffUpsampledNii = nib.load(saveDir + '/coefficientsUpsampledb' + str(bvalue) + 'n' + str(order) + '.nii.gz')
    coeffUpsampledData = coeffUpsampledNii.get_data()
    segUpsampledNii = nib.load(out + '/HCP_seg_clipped_new.nii.gz')
    segUpsampledData = segUpsampledNii.get_data()

    #csf_coeffs = coeffUpsampledData[(segUpsampledData[:,:,:,2] > 0.999)]

    #print csf_coeffs.shape
    #csf_coeffs_mean = np.mean(csf_coeffs,axis=0)
    #csf_coeffs_mean.shape
    #print csf_coeffs_mean
    
    #coeffNii = nib.load(saveDir + '/coefficientsb' + str(bvalue) + 'n' + str(order) + '.nii.gz')
    #csf_coeffs_rep = coeffNii.dataobj[84, 36, 66, :]
    #np.save('Files/SphericalHarmonics/csf_coeff_b'+str(bvalue),csf_coeffs_rep)
    csf_coeffs_rep = csf_factor * np.load('Files/SphericalHarmonics/csf_coeff_b'+str(bvalue)+'.npy')
    print(csf_coeffs_rep)

    for i in range(0, segUpsampledData.shape[0]):
        for j in range(0, segUpsampledData.shape[1]):
            for k in range(0, segUpsampledData.shape[2]):
                if segUpsampledData[i, j, k, 2] > 0.9:
                    coeffUpsampledData[i, j, k, :] = csf_coeffs_rep 
    coeffUpsampledNii.to_filename(saveDir + '/coefficientsUpsampledb' + str(bvalue) + 'n' + str(order) + '.nii.gz')
    #Free memory
    del coeffUpsampledData, segUpsampledData

