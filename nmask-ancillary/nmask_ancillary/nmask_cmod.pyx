# cython: language_level=3, boundscheck=False

import numpy as np


# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from libc.math cimport fabs

DTYPEINT = np.int
DTYPEFL = np.float32
DTYPEINT8 = np.uint8

ctypedef np.int_t DTYPEINT_t
ctypedef np.float32_t DTYPEFL_t
ctypedef np.uint8_t DTYPEINT8_t
ctypedef np.int16_t DTYPEINT16_t

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)


cdef float findnbsa( float* vsa, int k, int N, int vss, DTYPEINT8_t* dws_flags, DTYPEINT8_t* vtsmask):

    """

    Description: 

    Given the length of the segment is N, the function will find N clear pixels closest to the segment from the left, N
    N clear pixels closest to the segment from the right, calculate the mean of these 2N pixels

    Parameters:
    
    vsa: float, 1D array
        time series of mean of surface relectance values
    k: integer 
        location of the specified segment
    vss: integer
        length of the vsa
    N: integer
        length of the time series segment
    dws_flags: uint8, 1D array
        flags indicating that a pixel is either a non-shadow pixel or a water pixel
    vtsmask: uint8, 1D array
        time series of cloud/shadow labels
    
    
    Return:  mean values of neighbour pixels of the specified segment

 
    """

    # clear pixel counter
    cdef int cc = 0

    # Search direction, 0 -- search the left, 1 -- search the right
    cdef int dr = 0

    # Directional flags, 0 -- search can be continued, 1 -- search reach the boundary
    cdef DTYPEINT8_t[:] mvd = np.zeros(2, dtype=np.uint8)

    # location of the left of the segment
    cdef int lpt = k

    # location of the right of the segment
    cdef int rpt = k + N - 1

    # sum of the found clear pixels
    cdef float mid = 0.0

    while cc < 2 * N:

        # search the left
        if dr == 0:
            # Not reach the left boundary yet
            if mvd[0] == 0:
                while True:
                    # Move the left pointer to 1 pixel left
                    lpt -= 1
                    # reach the begining of the time series?
                    if lpt < 0:
                        # Yes, modify the directional flags, change the search directional
                        mvd[dr] = 1
                        dr = 1
                        break
                    elif vtsmask[lpt] == 1 and dws_flags[lpt] == 1:
                        # No, if the pixel is a clear pixels and the pixel is not a potenial shadow pxiels
                        # Add the value of the pixel to the sum of the found clear pixels
                        mid += vsa[lpt]
                        # update the clear pixel counter, change the search direction to right
                        cc += 1
                        dr = 1
                        break
            else:
                dr = 1
        else:
            # search the right
            if mvd[1] == 0:
                # Not reach the right boundary yet
                while True:
                    # Move the right pointer to 1 pixel right
                    rpt += 1
                    # reach the end of the time series?
                    if rpt == vss:
                        # Yes, modify the directional flags, change the srach directional
                        mvd[dr] = 1
                        dr = 0
                        break
                    elif vtsmask[rpt] == 1 and dws_flags[rpt] == 1:
                        # No, if the pixel is a clear pixels and the pixel is not a potenial shadow pxiels
                        # Add the value of the pixel to the sum of the found clear pixels
                        mid += vsa[rpt]
                        # update the clear pixel counter, change the search direction to left
                        cc += 1
                        dr = 0
                        break
            else:
                dr = 0

        # The search reach the boundaries in both direction, exit the search
        if mvd[0] == 1 and mvd[1] == 1:
            break

    # if not enough clear pixels found, return 0, otherwise return the mean of the found 2N clear pixels
    if cc < 2 * N:
        return 0
    else:
        return mid / (2 * N)

    

cdef testpair(float* sa, float* dwi, int N, DTYPEINT8_t* tsmask, int tn):

    """

    Function Name: testpair

    Description: 
    
    This function identifies cloud and shadow pixels in a time series by comparing its value to its neighbours
  
    Parameters:
    
    sa: float, 1D array
        time series of the mean of surface reflectance value of the 6 spectral bands
    dwi: float, 1D array, 
        time series of MNDWI (modified normalised water difference index)
    tsmasak: uint8, 1D array
        time series of cloud/shadow labels
    

    Return:

    None, tsmask is updated 
    
 
    """
    # cloud detection threshold, the lower the value, the higher cloud detection rate
    cdef float cspkthd = 0.42

    # shade detection threshold, the lower the value, the higher shade detection rate
    cdef float sspkthd = 0.42

    # the minimum theshold of a cloud pixel, i.e., all cloud pixels will have a band average
    # value higher that this theshold
    cdef float cloudthd = 0.10
    

    # The shadow pixel theshold
    cdef float shadowthd = 0.055
    
    #cdef int tn
    
    # length of the time series
    #tn=tsmask.size
    

    # Find all clear pixels in the time series
    validx = <int *> malloc(tn*sizeof(int))
    
    cdef int i, cc, vss, j, idx
    
    cc=0
    for i in range(tn):
        if tsmask[i]==1:
            validx[cc]=i
            cc=cc+1
        
    # Number of valid pixels in the time series
     
    vss=cc
    
    # Not enough clear pixels in the time series
    if vss < 3 * N:
        free(validx)
        return
    
    
    # Filter out invalid, cloud, shadow points in time series
    
    #cdef np.ndarray[DTYPEFL_t, ndim=1] vsa = sa[validx]
    vsa = <float *> malloc(vss*sizeof(float))
    
    #cdef np.ndarray[DTYPEFL_t, ndim=1] vdwi = dwi[validx]
    vdwi = <float *> malloc(vss*sizeof(float))
    
    #cdef np.ndarray[DTYPEINT8_t, ndim=1] vtsmask = tsmask[validx]
    vtsmask = <DTYPEINT8_t *> malloc(vss*sizeof(char))
   
    # flags which indicates if a segment in the time series had been checked 
    chmarker = <DTYPEINT8_t *> malloc(vss*sizeof(char))
    
    # flags which indicates a pixel is either a non-shadow or a water pixels
    dws_flags = <DTYPEINT8_t *> malloc(vss*sizeof(char))
    
    
    
    for i in range(vss):
        idx=validx[i]
        vsa[i]=sa[idx]
        vdwi[i]=dwi[idx]
        vtsmask[i]=tsmask[idx]
        chmarker[i]=0
        if vsa[i]>shadowthd or vdwi[i]>0:
            dws_flags[i]=1
        else:
            dws_flags[i]=0
        
        
    
    
    
    
    # Total number of segments in the time series
    cdef int numse = vss - N + 1

    # mean values of the time series segments
    msa = <float *> malloc(numse*sizeof(float))
    
    
    cdef float pms
    
    # calculate mean values of the time series segments
    for i in range(numse):
        if N == 1:
            msa[i]=vsa[i]
        else:
            pms=0
            for j in range(i, i+N):
                pms += vsa[j]
            msa[i]=pms/N
    
    
 

    # index array corresponding to descending order of the values of msa array
    sts = <int *> malloc(numse*sizeof(int))
   
    # index array corresponding to descending order of the values of msa array
    bfmsa = <float *> malloc(numse*sizeof(float))
    
    
    

    for i in range(numse):
        bfmsa[i] = msa[i]
        sts[i] = i
    
    cdef float tmp
    cdef int stmp
    
    
    
    # sort the time series of mean of the segemnts in descending order
    #upbound=1000.0
    for i in range(numse-1):
        
        for j in range(i+1, numse):
            #upbound = 100.0
            if (bfmsa[j]> bfmsa[i]):
                tmp = bfmsa[j]
                bfmsa[j] = bfmsa[i]
                bfmsa[i]=tmp
                stmp = sts[j]
                sts[j] = sts[i]
                sts[i] = stmp
        
    
  
  
    cdef int k
    cdef float m1, m2, mid
  

    
    # detect anormalty among time series data
    for i in range(numse):
        k=sts[i]
        if chmarker[k] == 0:
            # mean of the segment
            m2 = msa[k]
            # mean of the neighbouring 2N pixels
            mid = findnbsa(vsa, k, N, vss, dws_flags, vtsmask)

            # check if the mean of segemnt is significantly different from the neighbouring pixels
            if m2 > mid and mid > 0:
                if (m2 - mid) / mid > cspkthd and m2 > cloudthd:
                    # cloud pixels
                
                    for j in range(k, k+N):
                        vtsmask[j] = 2
                        chmarker[j] = 1

            elif mid > m2 and m2 > 0:
    
                if (mid - m2) / m2 > sspkthd and m2 < shadowthd:
                    # shadow pixels
                  
                    for j in range(k, k+N):
                        vtsmask[j] = 3
                        chmarker[j] = 1
                   
    # update the orginal time series mask with vtsmask
    
    for i in range(vss):
        tsmask[validx[i]] = vtsmask[i]
    
    
    # release memory allocated for the function 
    free(validx)
    free(vsa)
    free(vdwi)
    free(vtsmask)
    free(chmarker)
    free(dws_flags)
    free(msa)
    free(sts)
    free(bfmsa)



def spatial_filter_v2(onescene):

    """

    Function Name: 

    Description: 
    
    This function labels cloud and shadow pixels with less than M surrounding cloud/shadow pixels as clear pixels
  
    Parameters:
    
    onescene: uint8, 2D array
        One scene of the cloud mask dataarray
        
    Return:  
    
    updated cloud mask 2D array with updated cloud/shadow mask values
 
    """
    
    cdef int M = 2
    cdef int N = 8
    
    cdef int irow, icol, y, x, pnum, idx, cc, i, didx
    
    cdef int dy[8]
    cdef int dx[8]
    
    dy[:] = [-1, -1, -1, 0, 1, 1, 1, 0]
    dx[:] = [-1, 0, 1, 1, 1, 0, -1, -1]
    
    irow, icol = onescene.shape
    
    pnum = irow * icol
    
    tsmask = <DTYPEINT8_t *> malloc(pnum*sizeof(char))
    
    for y in range(irow):
        for x in range(icol):
            idx = y*icol + x
            tsmask[idx] = onescene[y, x]
    
    for y in range(1, irow-1):
        for x in range(1, icol-1):
            # if the center pixel in the block is a cloud or shadow
            idx = y * icol + x
            if (tsmask[idx] == 2 or tsmask[idx] == 3):
                # if total number of cloud/shadow pixels in the block is less than M+1,
                # label the center pixel as a clear pixel
                cc=0
                for i in range(N):
                    didx = (y+dy[i])*icol + x + dx[i]
                    if (tsmask[didx] ==2 or tsmask[didx]==3):
                        cc += 1
                    
                if (cc < M):
                    tsmask[idx] = 1
                    
    
    for y in range(irow):
        for x in range(icol):
            idx = y*icol + x
            onescene[y, x] = tsmask[idx] 
    
    free(tsmask)

    return onescene


def perpixel_filter_direct_core(DTYPEINT16_t[:] blue, DTYPEINT16_t[:] green, DTYPEINT16_t[:] red, DTYPEINT16_t[:] nir, DTYPEINT16_t[:] swir1, DTYPEINT16_t[:] swir2, DTYPEINT8_t[:] tsmask):
    """

    Description: 
    
    This function performs time series cloud/shadow detection for one pixel
  
    Parameters: 
    
    blue, green, red, nir, swir1, swir2: float, 1D arrays
        Surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for the pixel
        
    tsmask: uint8, 1D array
        Cloud /shadow mask time series for the pixel
    
    Return:  
    
    tsmask: uint8, 1D array
        Cloud /shadow mask time series for the pixel
    
    Updated cloud/shadow mask time serie

 
    """
    
    cdef int tn
   
    # length of the time series
    #tn = tsmask.size
    tn = blue.size
    
    sa = <float *> malloc(tn*sizeof(float))
    mndwi = <float *> malloc(tn*sizeof(float))
    msavi = <float *> malloc(tn*sizeof(float))
    wbi = <float *> malloc(tn*sizeof(float))
    rgm = <float *> malloc(tn*sizeof(float))
    grbm = <float *> malloc(tn*sizeof(float))
    ctsmask = <DTYPEINT8_t *> malloc(tn*sizeof(char))
    
    cdef float blue_t, green_t, red_t, nir_t, swir1_t, swir2_t
    cdef float scale = 10000.0
    cdef float ival = -999.0/scale
    
    cdef int i
    cdef float scom
    
    # Bright cloud theshold
    cdef float maxcldthd = 0.45
    
    for i in range(tn):
        ctsmask[i]=1
        blue_t = blue[i]/scale
        green_t = green[i]/scale
        red_t = red[i]/scale
        nir_t = nir[i]/scale
        swir1_t = swir1[i]/scale
        swir2_t = swir2[i]/scale
        #print(blue_t, green_t, red_t, nir_t, swir1_t, swir2_t)
        if (blue_t <=ival or green_t<=ival or red_t<=ival or nir_t<=ival or swir1_t<=ival or swir2_t<=ival or 
            blue_t == 0 or green_t == 0 or red_t==0 or nir_t ==0 or swir1_t==0 or swir2_t==0):
            ctsmask[i]=0
        else:
            sa[i] = (blue_t+green_t+red_t+nir_t+swir1_t+swir2_t)/6
            if (green_t + swir1_t !=0):
                mndwi[i] = ((green_t - swir1_t) / (green_t + swir1_t))
            else:
                ctsmask[i]=0
            scom = (2*nir_t+1)*(2*nir_t+1) - 8*(nir_t - red_t)
            if (scom > 0):
                msavi[i] = (2 * nir_t + 1 -sqrt(scom))/2
            else:
                ctsmask[i]=0
            
            wbi[i] = (red_t - blue_t) / blue_t
            rgm[i] = red_t + blue_t
            grbm[i] = (green_t - (red_t + blue_t) / 2) / ((red_t + blue_t) / 2)
                      
   
            # label all ultra-bright pixels as clouds
            if (sa[i] > maxcldthd):
                ctsmask[i] = 2
    
    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, ctsmask, tn)
    testpair(sa, mndwi, 1, ctsmask, tn)
    testpair(sa, mndwi, 1, ctsmask, tn)

    # detect 2 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 2, ctsmask, tn)
    testpair(sa, mndwi, 2, ctsmask, tn)

    # detect 3 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 3, ctsmask, tn)

    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, ctsmask, tn)

    # cloud shadow theshold
    cdef float shdthd = 0.05

    # mndwi water pixel theshold
    cdef float dwithd = -0.05

    # mndwi baregroud pixel theshold
    cdef float landcloudthd = -0.38

    # msavi water pixel theshold
    cdef float avithd = 0.06

    # mndwi water pixel theshold
    cdef float wtdthd = -0.2

    cdef DTYPEINT8_t lab
                      
    for i in range(tn):

        lab = ctsmask[i]
        if lab == 3 and mndwi[i] > dwithd and sa[i] < shdthd:  # water pixel, not shadow
            ctsmask[i] = 1
           

        if lab == 2 and mndwi[i] < landcloudthd:  # bare ground, not cloud
            ctsmask[i] = 1

        if (
            lab == 3 and msavi[i] < avithd and mndwi[i] > wtdthd
        ):  # water pixel, not shadow
            ctsmask[i] = 1

        if (
            lab == 1
            and wbi[i] < -0.02
            and rgm[i] > 0.06
            and rgm[i] < 0.29
            and mndwi[i] < -0.1
            and grbm[i] < 0.2
        ):  # thin cloud
            ctsmask[i] = 2

    #tsmask=np.zeros(tn, dtype=np.uint8)
    
    for i in range(tn):
        tsmask[i] = ctsmask[i]
            
    free(sa)
    free(mndwi)
    free(msavi)
    free(wbi)
    free(rgm)
    free(grbm)
    free(ctsmask)
    
    return tsmask

def perpixel_filter_indices_core(DTYPEINT16_t[:] blue, DTYPEINT16_t[:] green, DTYPEINT16_t[:] red, DTYPEINT16_t[:] nir, DTYPEINT16_t[:] swir1, DTYPEINT16_t[:] swir2):
  
    """

    Description: 
    
    This function calculate long term mean  of a set of spectral indices for a pixel
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for a pixel
        Data types and formats: int16, 1D arrays 
       
            
        
    
    Return:  
    
        Descriptions: long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
        Names of variables: indices
        Data types and formats: float32, 1D arrays 
        Order of indices: mu of s6m,  mu of mndwi,  mu of msavi,  mu of whi
    
    
          
 
    """
    
    cdef int tn
   
    # length of the time series
    #tn = tsmask.size
    tn = blue.size
    
    sa = <float *> malloc(tn*sizeof(float))
    mndwi = <float *> malloc(tn*sizeof(float))
    msavi = <float *> malloc(tn*sizeof(float))
    wbi = <float *> malloc(tn*sizeof(float))
    whi = <float *> malloc(tn*sizeof(float))
    rgm = <float *> malloc(tn*sizeof(float))
    grbm = <float *> malloc(tn*sizeof(float))
    ctsmask = <DTYPEINT8_t *> malloc(tn*sizeof(char))
    
    cdef float blue_t, green_t, red_t, nir_t, swir1_t, swir2_t
    cdef float scale = 10000.0
    cdef float ival = -999.0/scale
    
    cdef int i
    cdef int cc
    cdef float scom
    cdef float mv
    
    # Bright cloud theshold
    cdef float maxcldthd = 0.45
    
    for i in range(tn):
        ctsmask[i]=1
        blue_t = blue[i]/scale
        green_t = green[i]/scale
        red_t = red[i]/scale
        nir_t = nir[i]/scale
        swir1_t = swir1[i]/scale
        swir2_t = swir2[i]/scale
        #print(blue_t, green_t, red_t, nir_t, swir1_t, swir2_t)
        if (blue_t <=ival or green_t<=ival or red_t<=ival or nir_t<=ival or swir1_t<=ival or swir2_t<=ival or 
            blue_t == 0 or green_t == 0 or red_t==0 or nir_t ==0 or swir1_t==0 or swir2_t==0):
            ctsmask[i]=0
        else:
            sa[i] = (blue_t+green_t+red_t+nir_t+swir1_t+swir2_t)/6
            if (green_t + swir1_t !=0):
                mndwi[i] = ((green_t - swir1_t) / (green_t + swir1_t))
            else:
                ctsmask[i]=0
            scom = (2*nir_t+1)*(2*nir_t+1) - 8*(nir_t - red_t)
            if (scom > 0):
                msavi[i] = (2 * nir_t + 1 -sqrt(scom))/2
            else:
                ctsmask[i]=0
            
            wbi[i] = (red_t - blue_t) / blue_t
            rgm[i] = red_t + blue_t
            grbm[i] = (green_t - (red_t + blue_t) / 2) / ((red_t + blue_t) / 2)
                      
            mv = (green_t + red_t + blue_t) / 3
            whi[i] = 0
            whi[i] += fabs((blue_t - mv)/mv)
            whi[i] += fabs((green_t - mv)/mv)
            whi[i] += fabs((red_t - mv)/mv)

            # label all ultra-bright pixels as clouds
            if (sa[i] > maxcldthd):
                ctsmask[i] = 2
    
    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, ctsmask, tn)
    testpair(sa, mndwi, 1, ctsmask, tn)
    testpair(sa, mndwi, 1, ctsmask, tn)

    # detect 2 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 2, ctsmask, tn)
    testpair(sa, mndwi, 2, ctsmask, tn)

    # detect 3 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 3, ctsmask, tn)

    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, ctsmask, tn)

    # cloud shadow theshold
    cdef float shdthd = 0.05

    # mndwi water pixel theshold
    cdef float dwithd = -0.05

    # mndwi baregroud pixel theshold
    cdef float landcloudthd = -0.38

    # msavi water pixel theshold
    cdef float avithd = 0.06

    # mndwi water pixel theshold
    cdef float wtdthd = -0.2

    cdef DTYPEINT8_t lab
                      
    for i in range(tn):

        lab = ctsmask[i]
        if lab == 3 and mndwi[i] > dwithd and sa[i] < shdthd:  # water pixel, not shadow
            ctsmask[i] = 1
           

        if lab == 2 and mndwi[i] < landcloudthd:  # bare ground, not cloud
            ctsmask[i] = 1

        if (
            lab == 3 and msavi[i] < avithd and mndwi[i] > wtdthd
        ):  # water pixel, not shadow
            ctsmask[i] = 1

        if (
            lab == 1
            and wbi[i] < -0.02
            and rgm[i] > 0.06
            and rgm[i] < 0.29
            and mndwi[i] < -0.1
            and grbm[i] < 0.2
        ):  # thin cloud
            ctsmask[i] = 2

    cdef float sa_u, mndwi_u, msavi_u, whi_u
    
    
    sa_u = 0
    mndwi_u = 0
    msavi_u = 0
    whi_u = 0
    
    
    cc = 0
    for i in range(tn):
        
        if ctsmask[i] == 1:
            sa_u += sa[i]
            mndwi_u += mndwi[i]
            msavi_u += msavi[i]
            whi_u   += whi[i]
            cc += 1
            
       
    
    if cc > 0:
        
        sa_u /= cc
        mndwi_u /= cc
        msavi_u /= cc
        whi_u /= cc
    
    
    
    indices=np.zeros(4, dtype=np.float32)
    
    indices[0] = sa_u
    indices[1] = mndwi_u
    indices[2] = msavi_u
    indices[3] = whi_u
              
    free(sa)
    free(mndwi)
    free(msavi)
    free(wbi)
    free(rgm)
    free(grbm)
    free(ctsmask)
    free(whi)
    
    return indices

def perpixel_filter_indices_std_core(DTYPEINT16_t[:] blue, DTYPEINT16_t[:] green, DTYPEINT16_t[:] red, DTYPEINT16_t[:] nir, DTYPEINT16_t[:] swir1, DTYPEINT16_t[:] swir2):
   
   
    """

    Description: 
    
    This function calculate long term mean and standard deviations of a set of spectral indices for a pixel
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for a pixel
        Data types and formats: int16, 1D arrays 
       
            
        
    
    Return:  
    
        Descriptions: long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
        Names of variables: indices
        Data types and formats: float32, 1D arrays 
        Order of indices: mu of s6m, std of s6m, mu of mndwi, std of mndwi, mu of msavi, std of msavi, mu of whi, std of whi
    
    
          
 
    """
    
    cdef int tn
   
    # length of the time series
    #tn = tsmask.size
    tn = blue.size
    
    sa = <float *> malloc(tn*sizeof(float))
    mndwi = <float *> malloc(tn*sizeof(float))
    msavi = <float *> malloc(tn*sizeof(float))
    wbi = <float *> malloc(tn*sizeof(float))
    whi = <float *> malloc(tn*sizeof(float))
    rgm = <float *> malloc(tn*sizeof(float))
    grbm = <float *> malloc(tn*sizeof(float))
    ctsmask = <DTYPEINT8_t *> malloc(tn*sizeof(char))
    
    cdef float blue_t, green_t, red_t, nir_t, swir1_t, swir2_t
    cdef float scale = 10000.0
    cdef float ival = -999.0/scale
    
    cdef int i
    cdef int cc
    cdef float scom
    cdef float mv
    
    # Bright cloud theshold
    cdef float maxcldthd = 0.45
    
    for i in range(tn):
        ctsmask[i]=1
        blue_t = blue[i]/scale
        green_t = green[i]/scale
        red_t = red[i]/scale
        nir_t = nir[i]/scale
        swir1_t = swir1[i]/scale
        swir2_t = swir2[i]/scale
        #print(blue_t, green_t, red_t, nir_t, swir1_t, swir2_t)
        if (blue_t <=ival or green_t<=ival or red_t<=ival or nir_t<=ival or swir1_t<=ival or swir2_t<=ival or 
            blue_t == 0 or green_t == 0 or red_t==0 or nir_t ==0 or swir1_t==0 or swir2_t==0):
            ctsmask[i]=0
        else:
            sa[i] = (blue_t+green_t+red_t+nir_t+swir1_t+swir2_t)/6
            if (green_t + swir1_t !=0):
                mndwi[i] = ((green_t - swir1_t) / (green_t + swir1_t))
            else:
                ctsmask[i]=0
            scom = (2*nir_t+1)*(2*nir_t+1) - 8*(nir_t - red_t)
            if (scom > 0):
                msavi[i] = (2 * nir_t + 1 -sqrt(scom))/2
            else:
                ctsmask[i]=0
            
            wbi[i] = (red_t - blue_t) / blue_t
            rgm[i] = red_t + blue_t
            grbm[i] = (green_t - (red_t + blue_t) / 2) / ((red_t + blue_t) / 2)
                      
            mv = (green_t + red_t + blue_t) / 3
            whi[i]=0
            whi[i] += fabs((blue_t - mv)/mv)
            whi[i] += fabs((green_t - mv)/mv)
            whi[i] += fabs((red_t - mv)/mv)

            # label all ultra-bright pixels as clouds
            if (sa[i] > maxcldthd):
                ctsmask[i] = 2
    
    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, ctsmask, tn)
    testpair(sa, mndwi, 1, ctsmask, tn)
    testpair(sa, mndwi, 1, ctsmask, tn)

    # detect 2 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 2, ctsmask, tn)
    testpair(sa, mndwi, 2, ctsmask, tn)

    # detect 3 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 3, ctsmask, tn)

    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, ctsmask, tn)

    # cloud shadow theshold
    cdef float shdthd = 0.05

    # mndwi water pixel theshold
    cdef float dwithd = -0.05

    # mndwi baregroud pixel theshold
    cdef float landcloudthd = -0.38

    # msavi water pixel theshold
    cdef float avithd = 0.06

    # mndwi water pixel theshold
    cdef float wtdthd = -0.2

    cdef DTYPEINT8_t lab
                      
    for i in range(tn):

        lab = ctsmask[i]
        if lab == 3 and mndwi[i] > dwithd and sa[i] < shdthd:  # water pixel, not shadow
            ctsmask[i] = 1
           

        if lab == 2 and mndwi[i] < landcloudthd:  # bare ground, not cloud
            ctsmask[i] = 1

        if (
            lab == 3 and msavi[i] < avithd and mndwi[i] > wtdthd
        ):  # water pixel, not shadow
            ctsmask[i] = 1

        if (
            lab == 1
            and wbi[i] < -0.02
            and rgm[i] > 0.06
            and rgm[i] < 0.29
            and mndwi[i] < -0.1
            and grbm[i] < 0.2
        ):  # thin cloud
            ctsmask[i] = 2

   
    cdef float sa_u, mndwi_u, msavi_u, whi_u
    cdef float sa_std, mndwi_std, msavi_std, whi_std
    
    sa_u = 0
    mndwi_u = 0
    msavi_u = 0
    whi_u = 0
    
    sa_std = 0
    mndwi_std = 0
    msavi_std = 0
    whi_std = 0
    
    cc = 0
    for i in range(tn):
        
        if ctsmask[i] == 1:
            
            sa_u += sa[i]
            mndwi_u += mndwi[i]
            msavi_u += msavi[i]
            whi_u   += whi[i]
            cc += 1
            
    
    if cc > 0:
        
        sa_u /= cc
        mndwi_u /= cc
        msavi_u /= cc
        whi_u /= cc
    
    
    cdef int NM1
    
    NM1 = cc - 1
    
    if cc>1:
        for i in range(tn):
            if ctsmask[i] == 1:
                           
                sa_std += (sa[i] - sa_u) *  (sa[i] - sa_u) 
                mndwi_std += (mndwi[i] - mndwi_u) * (mndwi[i] - mndwi_u)
                msavi_std += (msavi[i] - msavi_u) * (msavi[i] - msavi_u)
                whi_std += (whi[i] - whi_u) * (whi[i] - whi_u)
    
        
        
        sa_std /= NM1
        mndwi_std /= NM1
        msavi_std /= NM1
        whi_std /= NM1
    
    
        sa_std = sqrt(sa_std)
        mndwi_std = sqrt(mndwi_std)
        msavi_std = sqrt(msavi_std)
        whi_std = sqrt(whi_std)
    
    indices=np.zeros(8, dtype=np.float32)
    
    indices[0] = sa_u
    indices[1] = sa_std
    indices[2] = mndwi_u
    indices[3] = mndwi_std
    indices[4] = msavi_u
    indices[5] = msavi_std
    indices[6] = whi_u
    indices[7] = whi_std
    
    
    
    
    free(sa)
    free(mndwi)
    free(msavi)
    free(wbi)
    free(rgm)
    free(grbm)
    free(ctsmask)
    free(whi)
    
    return indices


def tsmask(DTYPEINT16_t[:, :, :] blue, DTYPEINT16_t[:, :, :] green, DTYPEINT16_t[:, :, :] red, DTYPEINT16_t[:, :, :] nir, DTYPEINT16_t[:, :, :] swir1, DTYPEINT16_t[:, :, :] swir2):
    
    
    
    """

    Description: 
    
    This function create cloud mask time series for a 2D array of pixels using TSmask algorithm
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for a 2D array of pixels
        Data types and formats: int16, 3D arrays 
        Order of dimensions: time, y, x
            
        
    
    Return:  
    
        Descriptions: cloud mask time series for a 2D array of pixels
        Names of variables: tsmask
        Data types and formats: uint8, 3D arrays 
        Order of dimensions: time, y, x
    
    
        
   
    """
    
    
    
    cshp = blue.shape
    
    
    tn  = cshp[0]
    irow = cshp[1]
    icol = cshp[2]
    
    tsmask = np.zeros((tn, irow, icol), dtype = np.uint8)
    
 
    cdef int y, x
    
    for y in range(irow):
        for x in range(icol):
            
            tsmask[:, y, x ] = perpixel_filter_direct_core ( blue[:, y, x ],  green[:, y, x ],  red[:, y, x ],  nir[:, y, x ],  
                                                                swir1[:, y, x ],  swir2[:, y, x ],  tsmask[:, y, x ])
        
    
    
    return tsmask



def tsmask_lastdim(DTYPEINT16_t[:, :, :] blue, DTYPEINT16_t[:, :, :] green, DTYPEINT16_t[:, :, :] red, DTYPEINT16_t[:, :, :] nir, DTYPEINT16_t[:, :, :] swir1, DTYPEINT16_t[:, :, :] swir2):
    
    
    """

    Description: 
    
    This function calculate long term mean of a set of spectral indices for a 2D array of pixels
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for a 2D array of pixels
        Data types and formats: int16, 3D arrays 
        Order of dimensions: y, x, time
            
        
    
    Return:  
    
        Descriptions: long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
        Names of variables: indices
        Data types and formats: float32, 3D arrays 
        Order of dimensions: y, x, indices (mu of s6m,  mu of mndwi,  mu of msavi, mu of whi)
    
    
          
 
    """
    
    
    
    cshp = blue.shape
    
    
    irow = cshp[0]
    icol = cshp[1]
    tn  = cshp[2]
    
    indices = np.zeros((irow, icol, 4), dtype = np.float32)
    
 
    cdef int y, x
    
    for y in range(irow):
        for x in range(icol):
                    
            indices[y, x, :] = perpixel_filter_indices_core ( blue[y, x, : ],  green[y, x, : ],  red[y, x, : ],  nir[y, x, : ],  
                                                                swir1[y, x, : ],  swir2[y, x, : ])
        
            
    
    
    return indices






def tsmask_lastdim_std(DTYPEINT16_t[:, :, :] blue, DTYPEINT16_t[:, :, :] green, DTYPEINT16_t[:, :, :] red, DTYPEINT16_t[:, :, :] nir, DTYPEINT16_t[:, :, :] swir1, DTYPEINT16_t[:, :, :] swir2):
    
    """

    Description: 
    
    This function calculate long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for a 2D array of pixels
        Data types and formats: int16, 3D arrays 
        Order of dimensions: y, x, time
            
        
    
    Return:  
    
        Descriptions: long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
        Names of variables: indices
        Data types and formats: float32, 3D arrays 
        Order of dimensions: y, x, indices (mu of s6m, std of s6m, mu of mndwi, std of mndwi, mu of msavi, std of msavi, mu of whi, std of whi)
    
    
          
 
    """
    
    
    cshp = blue.shape
    irow = cshp[0]
    icol = cshp[1]
    tn  = cshp[2]
    
    indices = np.zeros((irow, icol, 8), dtype = np.float32)
    
 
    cdef int y, x
    
    for y in range(irow):
        for x in range(icol):
                    
            indices[y, x, :] = perpixel_filter_indices_std_core ( blue[y, x, : ],  green[y, x, : ],  red[y, x, : ],  nir[y, x, : ],  
                                                                swir1[y, x, : ],  swir2[y, x, : ])
        
            
    
    
    return indices



def tsmask_firstdim_std(DTYPEINT16_t[:, :, :] blue, DTYPEINT16_t[:, :, :] green, DTYPEINT16_t[:, :, :] red, DTYPEINT16_t[:, :, :] nir, DTYPEINT16_t[:, :, :] swir1, DTYPEINT16_t[:, :, :] swir2):
    
    
    
    """

    Description: 
    
    This function calculate long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for a 2D array of pixels
        Data types and formats: int16, 3D arrays 
        Order of dimensions: time, y, x
            
        
    
    Return:  
    
        Descriptions: long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
        Names of variables: indices
        Data types and formats: float32, 3D arrays 
        Order of dimensions: indices (mu of s6m, std of s6m, mu of mndwi, std of mndwi, mu of msavi, std of msavi, mu of whi, std of whi), y, x
    
    
        
   
    """
    
    
    cshp = blue.shape
    
        
    irow = cshp[1]
    icol = cshp[2]
    tn  = cshp[0]
    
    indices = np.zeros((8, irow, icol), dtype = np.float32)
    
 
    cdef int y, x
    
    for y in range(irow):
        for x in range(icol):
                    
            indices[:, y, x] = perpixel_filter_indices_std_core ( blue[:, y, x],  green[:, y, x],  red[:, y, x],  nir[:, y, x],  
                                                                swir1[:, y, x],  swir2[:, y, x])
        
            
    
    
    return indices



def perpixel_bg_indices_core(DTYPEINT16_t[:] blue, DTYPEINT16_t[:] green, DTYPEINT16_t[:] red, DTYPEINT16_t[:] nir, DTYPEINT16_t[:] swir1, DTYPEINT16_t[:] swir2, DTYPEINT8_t[:] tsmask):

    """

    Description: 
    
    This function calculate long term mean of a set of spectral indices for one pixel
    
    Parameters: 
    
    blue, green, red, nir, swir1, swir2: float, 1D arrays
        Surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for the pixel
        
    tsmask: uint8, 1D array
        Cloud/shadow mask time series for the pixel
    
    Return:  
    
    Descriptions: long term mean of a set of spectral indices for one pixel
    Names of variables: bgi
    Data types and formats: float32, 1D arrays 
    Order of indices: mu of s6m,  mu of mndwi,  mu of msavi, mu of whi, mu of whi, mu of rgbd

 
    """
    
    cdef int tn
   
    # length of the time series
    #tn = tsmask.size
    tn = blue.size
    
    
    
    cdef float blue_t, green_t, red_t, nir_t, swir1_t, swir2_t
    cdef float sa, mndwi, msavi, scom, whi, rgbd
    cdef float scale = 10000.0
   
    cdef int i, cc
    
    sa = 0
    mndwi = 0
    msavi = 0  
    whi = 0
    rgbd = 0
    
    cc = 0
   
    
    for i in range(tn):
        if tsmask[i]==1:
            blue_t = blue[i]/scale
            green_t = green[i]/scale
            red_t = red[i]/scale
            nir_t = nir[i]/scale
            swir1_t = swir1[i]/scale
            swir2_t = swir2[i]/scale
            
            sa += (blue_t+green_t+red_t+nir_t+swir1_t+swir2_t)/6
            mndwi += ((green_t - swir1_t) / (green_t + swir1_t))
            scom = (2*nir_t+1)*(2*nir_t+1) - 8*(nir_t - red_t)
            msavi += (2 * nir_t + 1 -sqrt(scom))/2
            mv = (green_t + red_t + blue_t) / 3
            whi += fabs((blue_t - mv)/mv)
            whi += fabs((green_t - mv)/mv)
            whi += fabs((red_t - mv)/mv)
            rgbd += fabs ((blue_t - green_t) / (blue_t + green_t))   
            rgbd += fabs ((blue_t - red_t) / (blue_t + red_t)) 
            rgbd += fabs ((red_t - green_t) / (red_t + green_t)) 
            cc += 1
            

    if cc>0:
        sa /= cc
        mndwi /= cc
        msavi /= cc
        whi /= cc
        rgbd /= cc
        
            
    bgi = np.array([msavi, mndwi, sa, whi, rgbd], dtype=np.float32)
    
        
    return bgi






def getipdata(DTYPEINT16_t[:,:] blue, DTYPEINT16_t[:,:] green, DTYPEINT16_t[:,:] red, DTYPEINT16_t[:,:] nir, DTYPEINT16_t[:,:] swir1, DTYPEINT16_t[:,:] swir2,  DTYPEFL_t[:,:] s6m,  DTYPEFL_t[:,:] mndwi,  DTYPEFL_t[:,:] msavi,  DTYPEFL_t[:,:] whi):
    
    """

    Description: 
    
    This function creates input data for the Nmask ANN model for a 2D array of pixels
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: surface reflectance data of band blue, green, red, nir, swir1, swir2 for a 2D array of pixels
        Data types and formats: int16, 2D arrays 
        
        Names of variables: s6m, mndwi, msavi, whi
        Descriptions: long term means of a set of spectral indices 
        Data types and formats: float32, 2D arrays 
        
        Order of dimensions: y, x
            
        
    
    Return:  
    
        Descriptions: input data for the Nmask ANN model for a 2D array of pixels
        Names of variables: ipdata
        Data types and formats: float32, 3D arrays 
        Order of dimensions: y, x, input features
    
    
        
   
    """
    
    
    
    
    cdef float blue_t, green_t, red_t, nir_t, swir1_t, swir2_t
    cdef float scom, mv, ivd
    cdef float scale = 10000.0
   
    cdef int i, cc
    
    cdef float s6m_b, s6m_t
    cdef float mndwi_b, mndwi_t
    cdef float msavi_b, msavi_t
    cdef float whi_b, whi_t    

 
    
    ivd = -0.0999
     
    
    cshp = blue.shape
       
    
    irow = cshp[0]
    icol = cshp[1]
    
    ipdata = np.ones((irow, icol, 13), dtype = np.float32 )
    
    
 
    cdef int y
    cdef int x
    
    for y in range(irow):
        for x in range(icol):
            
            blue_t = blue[y, x]/scale
            green_t = green[y, x]/scale
            red_t = red[y, x]/scale
            nir_t = nir[y, x]/scale
            swir1_t = swir1[y, x]/scale
            swir2_t = swir2[y, x]/scale
            s6m_b = s6m[y, x]
            mndwi_b = mndwi[y, x]
            msavi_b = msavi[y, x]
            whi_b = whi[y, x]
            
            
            
            scom = (2*nir_t+1)*(2*nir_t+1) - 8*(nir_t - red_t)
            mv = (green_t + red_t + blue_t) / 3
    

            if (blue_t<=ivd or green_t<=ivd or red_t<=ivd or swir1_t<=ivd or 
                swir2_t<=ivd or scom<0 or mv==0 or s6m_b == 0 or mndwi_b == 0 or msavi_b ==0 or whi_b ==0):
                ipdata[y, x, 12]=0
            else:
                ipdata[y, x, 12]=1
                s6m_t = (blue_t+green_t+red_t+nir_t+swir1_t+swir2_t)/6
                mndwi_t = ((green_t - swir1_t) / (green_t + swir1_t))
                msavi_t = (2 * nir_t + 1 -np.sqrt(scom))/2
                whi_t = fabs((blue_t - mv)/mv) + fabs((green_t - mv)/mv) + fabs((red_t - mv)/mv)
                
                
                ipdata[y , x, 0] = s6m_b
                ipdata[y , x, 1] = s6m_t
                ipdata[y , x, 2] = (s6m_t - s6m_b) / s6m_b
                
                
                ipdata[y , x, 3] = mndwi_b
                ipdata[y , x, 4] = mndwi_t
                ipdata[y , x, 5] = (mndwi_t - mndwi_b) / mndwi_b
                
                ipdata[y , x, 6] = msavi_b
                ipdata[y , x, 7] = msavi_t
                ipdata[y , x, 8] = (msavi_t - msavi_b) / msavi_b
                
                ipdata[y , x, 9] = whi_b
                ipdata[y , x, 10] = whi_t
                ipdata[y , x, 11] = (whi_t - whi_b) / whi_b
                
                
                
                
                
    return ipdata