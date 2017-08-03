Motion can be included in simulations by specififying a path to a set of POSSUM motion files using the `--motion_dir` flag in generateFileStructure.py: e.g. `--motion_dir Files/Motion/inter_volume_example/`.

The directory must contain files motion0.txt, motion1.txt, where motionN.txt describes the motion experienced by volume N during acquistion and is formatted like a POSSUM motion file (more information [here.](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/POSSUM/UserGuide#Motion_sequence))

The example included here specifies inter-volume motion for up to 72 volumes. Intra-volume motion (both between acquistion of successive slices and during slice aacquisiton) is also possible. Motion files can be generated with the makeMotionFiles helper script.