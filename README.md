This code enables the simulation of DW-MR data using the [POSSUM](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/POSSUM) simulator.

### Dependencies
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
* Python with [dipy](http://nipy.org/dipy/), [nibabel](http://nipy.org/nibabel/), numpy and scipy

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

### Computational requirements

* generateFileStructure.py is fairly memory hungry - it may require up to 12GB RAM. 
* runPossum.py is where the meat of the simulation happens, and is very CPU intensive. Under the hood, it calls the FSL command possumX. For large simulations (i.e. anything more than a few slices of the brain) I recommend running this on a cluster environment that has been set up for automatic self-submission of FSL jobs (more information [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/SGE%20submission%20FAQ)). This will automatically divide each simulation into a number of jobs (the number can be set using the --num_processors flag), submit each to the cluster and handle recombination into the final image volume. On local systems you can easily get POSSUM to take advantage of multiple cores using [this](https://github.com/neurolabusc/fsl_sub) - make sure you set the --num_processors flag equal to the number of available cores.

### Datasets

Some of the simulated datasets used in the journal papers referenced below can be downloaded [here.](https://www.nitrc.org/projects/diffusionsim/)

### References

If you use DW-POSSUM in your research, please cite both the POSSUM and DW-POSSUM papers:

- Ivana Drobnjak, David Gavaghan, Endre Süli, Joe Pitt-Francis and Mark Jenkinson. Development of a functional magnetic resonance imaging simulator for modeling realistic rigid-body motion artifacts, MRM 56 364–380, 2006.
- Mark S. Graham, Ivana Drobnjak and Hui Zhang. Realistic simulation of artefacts in diffusion MRI for validating post-processing correction techniques, NeuroImage 125, 1079-1094, 2016.

We have extended POSSUM to simulate spin-echo sequences, enabling (in combination with this codebase) the simulation of DW-MR data with susceptibility artefacts. The new POSSUM release is forthcoming, though some simulated DW-MR datasets with susceptibility artefacts have been made available [here.](https://www.nitrc.org/projects/diffusionsim/) The relevant reference for this work is:

- Mark S. Graham, Ivana Drobnjak, Mark Jenkinson and Hui Zhang. Quantitative assessment of the susceptibility artefact and its interaction with motion in diffusion MRI, PLoS ONE 12(10): e0185647, 2017.


