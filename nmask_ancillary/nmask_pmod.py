#!/usr/bin/env python
# coding: utf-8



import time
import rasterio
import numpy as np
import datacube
from datacube.utils.cog import write_cog

import nmask_cmod as cym




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



def output_ds_to_cog(bandsets, outbandnames, dirc, loc_str, s2_ds):

    """

    Description: 

    This function outputs a set of dataarrays in a dataset as cloud-optimised GeoTIFF files

    Parameters:
    
    bandsets: list of string
        specified the names of the dataarray in the dataset to be saved 
    
    dirc: string 
        the directory where the image files are save 

    loc_str: string 
        the name of location where the image data are from, the string will be part of the file names 
    
    s2_ds: Xarray dataset Object
        the dataset contains the dataarrays

    Return:  

    None
    
    """

    # Create a list of date strings from the time series of time in the dataset

    timebandnames = get_timebandnames(s2_ds)

    for bandname, outputname in zip(bandsets, outbandnames):
        banddata = s2_ds[bandname]

        for i in range(len(s2_ds.time)):

            #  date of the satellite image as part of the name of the GeoTIFF
            datestr = timebandnames[i]

            # Convert current time step into a `xarray.DataArray`
            singletimestamp_da = banddata[i]

            # Create output filename
            filename = dirc + "/" + loc_str + "_" + outputname + "_" + datestr + ".tif"

            # Write GeoTIFF
            write_cog(geo_im=singletimestamp_da, fname=filename, nodata=255, overwrite=True)



def summarise(scenes):
    
    """

    Description: 
    
    This function calculate long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
  
    Parameters: 
    
        Names of variables: scences
        Descriptions: xarray dataset with S2 nbart time series of band blue, green, red, nir_2, swir_2, swir_3 for a 2D array of pixels
        Data types and formats: int16, 3D arrays 
        Order of dimensions: time, y, x
            
        
    
    Return:  
    
        Descriptions: long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
        Names of variables: indices
        Data types and formats: float32, 3D arrays 
        Order of dimensions: indices (mu of s6m, std of s6m, mu of mndwi, std of mndwi, mu of msavi, std of msavi, mu of whi, std of whi), y, x
    
    
        
   
    """
    
    
    blue = scenes.nbart_blue.data
    green = scenes.nbart_green.data
    red = scenes.nbart_red.data
    nir = scenes.nbart_nir_2.data
    swir1 = scenes.nbart_swir_2.data
    swir2 = scenes.nbart_swir_3.data
  
    indices = cym.tsmask_firstdim_std(blue, green, red, nir, swir1, swir2)
    
    indices_list=[ 's6m', 's6m_std', 'mndwi', 'mndwi_std','msavi', 'msavi_std','whi', 'whi_std']
    
    
    for i, indname in enumerate(indices_list):
        
        scenes[indname] = scenes.nbart_blue[0]
        scenes[indname].data = indices[i]
    
   
    
    return scenes



