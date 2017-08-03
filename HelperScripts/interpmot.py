#!/usr/bin/env python
#MJ's code to interpolate possum motion file
import numpy as np
import sys
# debugger
import pdb
# pdb.set_trace()

def possum_interpmot(umot,motype,tr,trslc,nslices,nvols):
    # parse arguments
    # if len(sys.argv)<=7:
    #     print("Usage: %s <motion_type> <tr> <tr_slice> <nslices> <nvols> <custom_motion_file> <output_file>" % sys.argv[0])
    #     print("  motion_type = 0 for continuous; 1 for between slices ; 2 for between volumes")
    #     sys.exit(0)

    # motype=int(sys.argv[1])
    # tr=float(sys.argv[2])
    # trslc=float(sys.argv[3])
    # nslices=int(sys.argv[4])
    # nvols=int(sys.argv[5])


    # set up general timings (of volumes or slices)
    voltimes=[x*tr for x in range(nvols)]
    slicetimes=[x*trslc for x in range(nslices)]
    allslicetimes=[]
    for vt in voltimes:
        allslicetimes += [vt+x for x in slicetimes]

    # choose appropriate set of times    
    if motype==0:  # continuous
        times=allslicetimes
    elif motype==1:  # between slices
        times=allslicetimes
    elif motype==2:  # between volumes
        times=voltimes
    else:
        print("Unknown option for motion type")
        sys.exit(1)

    #print(times)

    # load motion array from file and initialise variables
    #umot=np.loadtxt(fname)
    unt=umot.shape[0]
    ncols=umot.shape[1]
    imot=np.zeros([len(times),ncols])
            
    # interpolate values at new time points
    nupper=min(1,unt-1)  # upper bound
    for tidx in range(len(times)):
        t = times[tidx]
        while nupper<unt and umot[nupper][0]<t:
            nupper+=1
        nupper=min(nupper,unt-1)  # clamp to upper bound
        nlower=max(nupper-1,0)    # clamp to lower bound
        #print("nlower is %f, nupper is %f and unt is %f" % (nlower,nupper,unt))
        #print("Time is %f which should be in [%f,%f]" % (t,umot[nlower][0],umot[nupper][0]))
        dt = t-umot[nlower][0]
        tspan = float(umot[nupper][0] - umot[nlower][0])
        for col in range(ncols):
            val = (1.0-dt/tspan) * umot[nlower][col] + dt/tspan * umot[nupper][col]
            imot[tidx,col] = val
            #print("Val is %f which should be in [%f,%f]" % (val,umot[nlower][col],umot[nupper][col]))

    # add extra time points just prior to existing ones to create sharp transitions and flat periods       
    smalldt=0.000010        
    if motype==0:  # continuous (nothing else to do)
        newimot=imot
    else:     # put in extra timepoints to get flat epochs and sharp transitions
        newimot=np.zeros([len(times)*2,ncols])
        imidx=0
        for tidx in range(len(times)):
            newimot[imidx,:]=imot[tidx,:]
            imidx+=1
            if tidx+1<len(times):
                newimot[imidx,0]=imot[tidx+1,0]-smalldt
            else:
                newimot[imidx,0]=imot[tidx,0]+smalldt
            newimot[imidx,1:]=imot[tidx,1:]
            imidx+=1
    return newimot
    # output interpolated results            
    #print(newimot)
    #print(newimot.shape)
    np.savetxt(oname, newimot)



