#!/bin/bash

#Download files from dropbox
curl -L -o possumtest.zip https://www.dropbox.com/sh/9i6bmpgpjcx32c8/AADddUGI_P3wPf_GNrhGOidPa?dl=1
curl -L -o segmentations.zip https://www.dropbox.com/sh/44hxzb54r4g360t/AAD9iR9AanaMEYtIrxLoVhE6a?dl=1
curl -L -o coefficientsUpsampledb1000n8.nii.gz https://www.dropbox.com/s/8jxvo081m1ahfsw/coefficientsUpsampledb1000n8.nii.gz?dl=1
curl -L -o coefficientsUpsampledb2000n8.nii.gz https://www.dropbox.com/s/4mpl9csws3ln1sd/coefficientsUpsampledb2000n8.nii.gz?dl=1


#Move to correct location
mkdir Files
mkdir Files/SphericalHarmonics
unzip -nx -d Files/POSSUMdirectories possumtest.zip -x /
unzip -nx -d Files/Segmentations segmentations.zip -x /
mv coefficientsUpsampledb1000n8.nii.gz Files/SphericalHarmonics/
mv coefficientsUpsampledb2000n8.nii.gz Files/SphericalHarmonics/


#Clean up
rm possumtest.zip
rm segmentations.zip

