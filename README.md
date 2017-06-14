This code enables the simulation of DW-MR data using the POSSUM simulator.

###Setup
1. Make sure you have [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) installed 
2. Clone this repository 
3. Run `bash getFiles.sh` to download the imaging data


###Tutorial
DW-POSSUM is run in three stages. First all the files required for the simulation are generated (generateFileStructure.py). Then the simulations are run (runPossum.py). Finally everything is tidied up (postProcess.py).

1. Generate required files 
```bash
python generatedFileStructure.py
```

2. Run POSSUM
```bash
python runPossum.py
```

3. Tidy up
```bash
python postProcess.py
```

