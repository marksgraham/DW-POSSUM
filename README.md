This code enables the simulation of DW-MR data using the POSSUM simulator.

### Setup
1. Make sure you have [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) installed 
2. Clone this repository 
3. Run `bash getFiles.sh` to download the imaging data


### Tutorial
DW-POSSUM is run in three stages. First all the files required for the simulation are generated (generateFileStructure.py). Then the simulations are run (runPossum.py). Finally everything is tidied up (postProcess.py).

1. Generate required files 
```bash
python generateFileStructure.py Files/POSSUMdirectories/possumSimdirOneSlice/ Test/ Files/Bvalsbvecs/bvalsfmrib Files/Bvalsbvecs/bvecsfmrib --num_images 3
```

2. Run POSSUM
```bash
python runPossum.py Test 3
```

3. Tidy up
```bash
postProcess.py Test 3 --noise_levels 0 0.0081 0.0165
```
Your simulations will be stored as 4D nifti files in Test/Results/


