#!/bin/bash

if [ $# -eq 0 ]
	then

	#Download files from dropbox
	curl -L -o possumtest.zip https://www.dropbox.com/sh/xk7kclzpwhe6h90/AADPg_57AG3VbJ7y3cco6zTya?dl=1
	curl -L -o HCP_seg_clipped.nii.gz https://www.dropbox.com/s/q9esf1vbazkv3q9/HCP_seg_clipped.nii.gz?dl=1
	curl -L -o coefficientsUpsampledb1000n8.nii.gz https://www.dropbox.com/s/8jxvo081m1ahfsw/coefficientsUpsampledb1000n8.nii.gz?dl=1
	curl -L -o coefficientsUpsampledb2000n8.nii.gz https://www.dropbox.com/s/4mpl9csws3ln1sd/coefficientsUpsampledb2000n8.nii.gz?dl=1
	curl -L -o csf_coeff_b1000.npy https://www.dropbox.com/s/let79ssf2odr7mc/csf_coeff_b1000.npy?dl=1
	curl -L -o csf_coeff_b2000.npy https://www.dropbox.com/s/ikfdevqtf1yy7si/csf_coeff_b2000.npy?dl=1


	#Move to correct location
	mkdir Files/SphericalHarmonics
	mkdir Files/Segmentations
	unzip -nx -d Files/POSSUMdirectories/ possumtest.zip -x /
	mv  HCP_seg_clipped.nii.gz  Files/Segmentations/
	mv coefficientsUpsampledb1000n8.nii.gz Files/SphericalHarmonics/
	mv coefficientsUpsampledb2000n8.nii.gz Files/SphericalHarmonics/
	mv csf_coeff_b1000.npy Files/SphericalHarmonics/
	mv csf_coeff_b2000.npy Files/SphericalHarmonics/

fi

if [ $# -eq 1 ]
	then
	echo 'Getting low-resolution files'
	#Download files from dropbox
	curl -L -o possumtest.zip https://www.dropbox.com/sh/xk7kclzpwhe6h90/AADPg_57AG3VbJ7y3cco6zTya?dl=1
	curl -L -o HCP_seg_clipped.nii.gz https://www.dropbox.com/s/2qxebk30pfivhwk/HCP_seg_clipped_diffusionspace.nii.gz?dl=0
	curl -L -o coefficientsb1000n8.nii.gz https://www.dropbox.com/s/5edkiw4chz04r7e/coefficientsb1000n8.nii.gz?dl=1
	curl -L -o coefficientsb2000n8.nii.gz https://www.dropbox.com/s/lgn5qbkr9fs0zmo/coefficientsb2000n8.nii.gz?dl=1


	#Move to correct location
	mkdir Files/SphericalHarmonics
	mkdir Files/Segmentations
	unzip -nx -d Files/POSSUMdirectories/ possumtest.zip -x /
	mv  HCP_seg_clipped.nii.gz  Files/Segmentations/
	mv coefficientsb1000n8.nii.gz Files/SphericalHarmonics/coefficientsUpsampledb1000n8.nii.gz
	mv coefficientsb2000n8.nii.gz Files/SphericalHarmonics/coefficientsUpsampledb2000n8.nii.gz

fi


#Clean up
rm possumtest.zip

