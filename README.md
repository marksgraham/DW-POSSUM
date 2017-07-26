This code enables the simulation of DW-MR data using the POSSUM simulator.

### Dependencies
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
* Python 2.x with [dipy](http://nipy.org/dipy/), [nibabel](http://nipy.org/nibabel/), numpy and scipy

### Setup
1. Clone this repository 
2. Run `bash getFiles.sh` to download the imaging data


### Quick-start tutorial
DW-POSSUM is run in three stages. First all the files required for the simulation are generated (generateFileStructure.py). Then the simulations are run (runPossum.py). Finally everything is tidied up (postProcess.py). The following commands will simulate the acquistion of a single slice for the first three entries in the bval/bvec file supplied (two b=0 and one b=1000 volumes).

1. Generate required files 
```bash
./generateFileStructure.py Files/POSSUMdirectories/possumSimdirOneSlice/ Test/ Files/Bvalsbvecs/bvalsfmrib Files/Bvalsbvecs/bvecsfmrib --num_images 3
```

2. Run POSSUM
```bash
./runPossum.py Test 3
```

3. Tidy up
```bash
./postProcess.py Test 3 --noise_levels 0 0.0081 0.0165
```
Your simulations will be stored as 4D nifti files in Test/Results/. 

Documentation for each command can be found using the -h flag. 

### References

If you use DW-POSSUM in your research, please cite both the POSSUM and DW-POSSUM papers:

- Ivana Drobnjak, David Gavaghan, Endre Süli, Joe Pitt-Francis and Mark Jenkinson. Development of a functional magnetic resonance imaging simulator for modeling realistic rigid-body motion artifacts, MRM 56 364–380, 2006.
- Mark S. Graham, Ivana Drobnjak and Hui Zhang. Realistic simulation of artefacts in diffusion MRI for validating post-processing correction techniques, NeuroImage 125, 1079-1094, 2016.


