#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import DEADataHandling
import numpy as np
import time
import rasterio
from spectral import envi


def load_s2_nbart_ts(
    dc, lat_top, lat_bottom, lon_left, lon_right, start_of_epoch, end_of_epoch
):

    """ 
    
    Function Name: load_s2_nbart_ts

    Description: 

    This function loads Sentinel-2 surface reflectance data from DEA database. 
    The spatial coverage is specfied by a rectangle bounding box. 
    The temporal coverage is specified by the start date and the end date of the time series data

    Parameters:
    
    dc: datacube Datacube object
        dc specifies which Datacube to be connected to load the data
    
    lat_top, lat_bottom, lon_left, lon_right: float32
        the lat./Lon. of the top-left and bottom-right corners of a rectangle bounding box
    
    start_of_epoch, end_of_epoch: date string
        the start date and the end date of the time series data 

    Return:   an Xarray contains 6 spectral bands of time series surface reflectance data 

   

    """

    # Define spatial and temporal coverage

    newquery = {
        "x": (lon_left, lon_right),
        "y": (lat_top, lat_bottom),
        "time": (start_of_epoch, end_of_epoch),
        "output_crs": "EPSG:3577",
        "resolution": (-20, 20),
    }

    # Names of targeted spectral bands

    allbands = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3",
    ]

    # Band names used with in the dataset
    new_bandlabels = ["blue", "green", "red", "nir", "swir1", "swir2"]

    # Load S2 data using Datacube API
    s2_ds = DEADataHandling.load_clearsentinel2(
        dc=dc,
        query=newquery,
        sensors=("s2a", "s2b"),
        product="ard",
        bands_of_interest=allbands,
        mask_pixel_quality=False,
        mask_invalid_data=False,
        masked_prop=0.0,
    )

    # Rename spectral band names to new band labels

    rndic = dict(zip(allbands, new_bandlabels))
    s2_ds = s2_ds.rename(rndic)

    # Add tsmask dataarray to the dataset

    s2_ds["tsmask"] = s2_ds["blue"]
    s2_ds["tsmask"].values = np.zeros(
        (s2_ds["time"].size, s2_ds["y"].size, s2_ds["x"].size), dtype=np.uint8
    )

    return s2_ds


# In[ ]:


def findnbsa(vsa, k, N, vss, dws_flags, vtsmask):

    """

    Function Name: findnbsa

    Description: 

    Given the length of the segment is N, the function will find N clear pixels closest to the segment from the left, N
    N clear pixels closest to the segment from the right, calculate the mean of these 2N pixels

    Parameters:
    
    vsa: float, 1D array
        time series of mean of surface relectance values
    k: integer 
        location of the specified segment
    N: integer
        length of the time series segment
    dws_flags: uint8, 1D array
        flags indicating that a pixel is either a non-shadow pixel or a water pixel
    vtsmask: uint8, 1D array
        time series of cloud/shadow labels
    
    
    Return:  mean values of neighbour pixels of the specified segment

 
    """

    # clear pixel counter
    cc = 0

    # Search direction, 0 -- search the left, 1 -- search the right
    dr = 0

    # Directional flags, 0 -- search can be continued, 1 -- search reach the boundary
    mvd = [0, 0]

    # location of the left of the segment
    lpt = k

    # location of the right of the segment
    rpt = k + N - 1

    # sum of the found clear pixels
    mid = 0.0

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
                        # Yes, modify the directional flags, change the srach directional
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


# In[ ]:


## Function F7


def perpixel_filter_direct(blue, green, red, nir, swir1, swir2, tsmask):

    """

    Function Name: perpixel_filter_direct

    Description: 
    
    This function performs time series cloud/shadow detection for one pixel
  
    Parameters: 
    
    blue, green, red, nir, swir1, swir2: float, 1D arrays
        Surface reflectance time series data of band blue, green, red, nir, swir1, swir2 for the pixel
        
    tsmask: float, 1D array
        Cloud /shadow mask time series for the pixel
    
    Return:  
    
    Updated cloud/shadow mask time serie

 
    """

    # scale factor
    scale = 10000.0

    # invalid value
    ivd = -999 / scale

    # initialise tsmask, all as clear pixels
    tsmask[:] = 1

    # copy and covert surface reflectance data as float
    blue = blue.copy().astype(np.float32) / scale
    green = green.copy().astype(np.float32) / scale
    red = red.copy().astype(np.float32) / scale
    nir = nir.copy().astype(np.float32) / scale
    swir1 = swir1.copy().astype(np.float32) / scale
    swir2 = swir2.copy().astype(np.float32) / scale

    # detect pixels with invalid data value
    tsmask[blue == ivd] = 0
    tsmask[green == ivd] = 0
    tsmask[red == ivd] = 0
    tsmask[nir == ivd] = 0
    tsmask[swir1 == ivd] = 0
    tsmask[swir2 == ivd] = 0

    # calculate indices

    # mean of 6 spectral bands
    sa = (blue + green + red + nir + swir1 + swir2) / 6

    # modified normalised difference water index
    mndwi = (green - swir1) / (green + swir1)

    # modified soil adjusted vegetation index
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) * (2 * nir + 1) - 8 * (nir - red))) / 2

    # Band different ratio between red and blue
    wbi = (red - blue) / blue

    # Sum of red and blue band
    rgm = red + blue

    # Band different ratio between green and mean of red and blue
    grbm = (green - (red + blue) / 2) / ((red + blue) / 2)

    # Bright cloud theshold
    maxcldthd = 0.45

    # label all ultra-bright pixels as clouds
    tsmask[sa > maxcldthd] = 2

    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, tsmask)
    testpair(sa, mndwi, 1, tsmask)
    testpair(sa, mndwi, 1, tsmask)

    # detect 2 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 2, tsmask)
    testpair(sa, mndwi, 2, tsmask)

    # detect 3 consecutive cloud / shadow pixels
    testpair(sa, mndwi, 3, tsmask)

    # detect single cloud / shadow pixels
    testpair(sa, mndwi, 1, tsmask)

    # cloud shadow theshold
    shdthd = 0.05

    # mndwi water pixel theshold
    dwithd = -0.05

    # mndwi baregroud pixel theshold
    landcloudthd = -0.38

    # msavi water pixel theshold
    avithd = 0.06

    # mndwi water pixel theshold
    wtdthd = -0.2

    for i, lab in enumerate(tsmask):

        if lab == 3 and mndwi[i] > dwithd and sa[i] < shdthd:  # water pixel, not shadow
            tsmask[i] = 1

        if lab == 2 and mndwi[i] < landcloudthd:  # bare ground, not cloud
            tsmask[i] = 1

        if (
            lab == 3 and msavi[i] < avithd and mndwi[i] > wtdthd
        ):  # water pixel, not shadow
            tsmask[i] = 1

        if (
            lab == 1
            and wbi[i] < -0.02
            and rgm[i] > 0.06
            and rgm[i] < 0.29
            and mndwi[i] < -0.1
            and grbm[i] < 0.2
        ):  # thin cloud
            tsmask[i] = 2

    return tsmask


# In[ ]:


# Function F8


def create_ts_tuples(s2_ds):

    """

    Function Name: create_ts_tuples

    Description: 
    
    This function creates a list of tuples of Sentinel-2 surface reflectance data, the list will 
    serve as the input when the Multiprocessing Pool method is called 

  
    Parameters: 
    
    s2_ds: Xarray dataset Object
        the dataset containing dataarrays of time series surface reflectance data
        
    Return: 
    
    a list of tuples of Sentinel-2 surface reflectance data

 
    """
    # number of rows
    irow = s2_ds["y"].size

    # number of columns
    icol = s2_ds["x"].size

    # total number of pixels
    pnum = irow * icol

    ts_tuples = []

    for i in np.arange(pnum):

        y = int(i / icol)
        x = i % icol

        # copy time series spectral data from the data set, scale the data to float32, in range (0, 1.0)

        blue = s2_ds["blue"].values[:, y, x]
        green = s2_ds["green"].values[:, y, x]
        red = s2_ds["red"].values[:, y, x]
        nir = s2_ds["nir"].values[:, y, x]
        swir1 = s2_ds["swir1"].values[:, y, x]
        swir2 = s2_ds["swir2"].values[:, y, x]
        tsmask = s2_ds["tsmask"].values[:, y, x]

        ts_tuples.append((blue, green, red, nir, swir1, swir2, tsmask))

    return ts_tuples


# In[ ]:


def write_multi_time_dataarray_v2(filename, dataarray, xs, ys, **profile_override):

    """

    Function Name: write_multi_time_dataarray_v2

    Description: 
    
    This function outputs an dataarray as an ENVI image
  
    Parameters:
    
    filename: string 
        The output filename including directory
        
    dataarray: dataarray
        The output dataarray 
    
    xs: integer
        number of columns 
        
    ys: integer
        hnumber of rows 
        
    profile_override: string
        profile name of output format

    Return:  
    
    None

 
    """

    profile = {
        "width": xs,
        "height": ys,
        "transform": dataarray.affine,
        "crs": "EPSG:3577",
        "count": len(dataarray.time),
        "dtype": str(dataarray.dtype),
    }
    profile.update(profile_override)

    with rasterio.open(str(filename), "w", **profile) as dest:
        for time_idx in range(len(dataarray.time)):
            bandnum = time_idx + 1
            dest.write(dataarray.isel(time=time_idx).data, bandnum)


# In[ ]:


def get_timebandnames(s2_ds):

    """

    Function Name: get_timebandnames

    Description: 
    
    This function creates a list of date strings from the time series of date and time in the dataset
  
    Parameters:
    
    s2_ds: Xarray dataset Object
        the dataset containing time series of date and time data
    
    Return:  
    
    a list of date strings
 
    """

    timelist = []
    for t in s2_ds["time"]:
        timelist.append(
            time.strftime("%Y-%m-%d", time.gmtime(t.astype(int) / 1000000000))
        )

    return timelist


# In[ ]:


def output_ds_to_ENVI(bandsets, outbandnames, dirc, s2_ds):

    """

    Function Name: output_ds_to_ENVI

    Description: 

    This function outputs a set of dataarrays in a dataset as ENVI images

    Parameters:
    
    bandsets: list of string
        specified the names of the dataarray in the dataset to be saved 
    
    dirc: string 
        the directory where the image files are save 
    
    s2_ds: Xarray dataset Object
        the dataset contains the dataarrays

    Return:  

    None
    
    """

    # Create a list of date strings from the time series of time in the dataset

    timebandnames = get_timebandnames(s2_ds)

    xs = s2_ds["x"].size
    ys = s2_ds["y"].size

    for bandname, outputname in zip(bandsets, outbandnames):
        banddata = s2_ds[bandname]
        filename = dirc + "/NBAR_" + outputname + ".img"

        # output dataarray as ENVI image file

        write_multi_time_dataarray_v2(filename, banddata, xs, ys, driver="ENVI")

        # Update ENVI header files with  the list of the date strings
        hdrfilename = dirc + "/NBAR_" + outputname + ".hdr"
        h = envi.read_envi_header(hdrfilename)
        h["band names"] = timebandnames
        envi.write_envi_header(hdrfilename, h)


# In[ ]:


def spatial_filter(onescene):

    """

    Function Name: 

    Description: 
    
    This function labels cloud and shadow pixels with less than M surrounding cloud/shadow pixels as clear pixels
  
    Parameters:
    
    onescene: uint8, 2D array
        One scene of the tsmask dataarray
        
    Return:  
    
    updated tsmask with cloud/shadow mask values
 
    """
    tsmask = onescene
    M = 2

    irow, icol = onescene.shape
    for y in np.arange(irow - 2):
        for x in np.arange(icol - 2):
            block = tsmask[y : y + 3, x : x + 3]
            # if the center pixel in the block is a cloud or shadow
            if block[1, 1] == 2 or block[1, 1] == 3:
                # if total number of cloud/shadow pixels in the block is less than M+1,
                # label the center pixel as a clear pixel

                if np.logical_or(block == 2, block == 3).sum() < M + 1:
                    tsmask[y + 1, x + 1] = 1

    return tsmask


# In[ ]:


def spatial_buffer(onescene):

    """

    Function Name: 

    Description: 
    
    This function labels neighbouring pixels of cloud and shadow pixels as cloud/shadow pixels
  
    Parameters:
    
    onescene: uint8, 2D array
        One scene of the tsmask dataarray
        
    Return:  

    updated tsmask with cloud/shadow mask values
 
    """

    tsmask = onescene
    newmask = onescene.copy()

    irow, icol = onescene.shape

    for y in np.arange(irow - 2):
        for x in np.arange(icol - 2):
            block = tsmask[y : y + 3, x : x + 3]
            # if the center pixel in the block is a cloud or shadow
            if block[1, 1] == 2 or block[1, 1] == 3:
                # label neighbouring pixels as cloud/shadow pixels
                newmask[y : y + 3, x : x + 3] = block[1, 1]

    # Only change clear pixels' labels
    vidx = tsmask == 1
    tsmask[vidx] = newmask[vidx]

    return tsmask


# In[ ]:


def testpair(sa, dwi, N, tsmask):

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
    cspkthd = 0.42

    # shade detection threshold, the lower the value, the higher shade detection rate
    sspkthd = 0.42

    # the minimum theshold of a cloud pixel, i.e., all cloud pixels will have a band average
    # value higher that this theshold
    cloudthd = 0.10

    # The shadow pixel theshold
    shadowthd = 0.055

    # Find all clear pixels in the time series
    validx = np.where(tsmask == 1)[0]

    # The number of the clear pixels in the time series
    vss = validx.size

    # Not enough clear pixels in the time series
    if vss < 3 * N:
        return

    # Filter out invalid, cloud, shadow points in time series
    vsa = sa[validx]
    vdwi = dwi[validx]
    vtsmask = tsmask[validx]

    # flags which indicates if
    chmarker = np.zeros(vss, dtype=np.int8)

    # flags which indicates a pixel is either a non-shadow or a water pixels
    dws_flags = np.logical_or(vsa > shadowthd, vdwi > 0)

    # Total number of segments in the time series
    numse = vss - N + 1

    # array to store mean of the segments
    msa = np.zeros(numse, dtype=np.float32)

    # calculate mean values of the time series segments

    if N == 1:
        msa = vsa

    else:
        for i in np.arange(numse):
            msa[i] = vsa[i : i + N].sum() / N

    # sort the time series of mean of the segemnts
    sts = np.argsort(msa)

    # reverse the order from ascending to descending, so that sts contains index number of msa array, from
    # highest values to the lowest
    sts = sts[::-1]

    for k in sts:

        if chmarker[k] == 0:
            # mean of the segment
            m2 = msa[k]
            # mean of the neighbouring 2N pixels
            mid = findnbsa(vsa, k, N, vss, dws_flags, vtsmask)

            # check if the mean of segemnt is significantly different from the neighbouring pixels
            if m2 > mid and mid > 0:
                if (m2 - mid) / mid > cspkthd and m2 > cloudthd:
                    # cloud pixels
                    vtsmask[k : k + N] = 2
                    chmarker[k : k + N] = 1

            elif mid > m2 and mid > 0:
                if (mid - m2) / m2 > sspkthd and m2 < shadowthd:
                    # shadow pixels
                    vtsmask[k : k + N] = 3
                    chmarker[k : k + N] = 1

    # update the orginal time series mask
    tsmask[validx] = vtsmask
