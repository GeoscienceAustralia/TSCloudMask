#!/usr/bin/env python
# coding: utf-8



import time
import xarray as xr
import rasterio
import numpy as np
import datacube
from datacube.utils.cog import write_cog

import nmask_cmod as cym
from dask.distributed import Client



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


def bs_tsmask(blue, green, red, nir, swir1, swir2):
  
    bsmask = cym.tsmask_lastdim_std(blue, green, red, nir, swir1, swir2)
    
    return bsmask



def gen_tsmask(chblue, chgreen, chred, chnir, chswir1, chswir2, chn):
    return xr.apply_ufunc(
        bs_tsmask, chblue, chgreen, chred, chnir, chswir1, chswir2,
        dask='parallelized',
        input_core_dims=[["time"], ["time"],["time"], ["time"],["time"], ["time"]],
        output_core_dims= [['indices']], 
        dask_gufunc_kwargs = {'output_sizes' : {'indices' : chn}},
        output_dtypes = [np.float32]
 
    )




def summarise_dask(scenes, ncpu, blocklist):
    
    """

    Description: 
    
    This function calculate long term mean and standard deviations of a set of spectral indices for a 2D array of pixels using a local dask client
  
    Parameters: 
    
        Names of variables: scences
        Descriptions: dask xarray dataset with S2 nbart time series of band blue, green, red, nir_2, swir_2, swir_3 for a 2D array of pixels, with 
        Data types and formats: int16, 3D arrays 
        Order of dimensions: time, y, x
        
        ncpu: int, number of cpu cores
        blocklist: int, a list of 2d bounding boxes, which are in form of [row1, row2, col1, col2]
        
    
    Return:  
    
        Descriptions: long term mean and standard deviations of a set of spectral indices for a 2D array of pixels
        Names of variables: indices
        Data types and formats: float32, 3D arrays 
        Order of dimensions: indices (mu of s6m, std of s6m, mu of mndwi, std of mndwi, mu of msavi, std of msavi, mu of whi, std of whi), y, x
    
    
        
   
    """
    
    
    client = Client(n_workers = ncpu, threads_per_worker=2, processes = True)
    

    indices_list=[ 's6m', 's6m_std', 'mndwi', 'mndwi_std','msavi', 'msavi_std','whi', 'whi_std']
    n_ids = len(indices_list)
    
    # number of rows
    irow=scenes['y'].size
    # number of columns
    icol=scenes['x'].size

    indices = np.zeros((irow, icol, n_ids), dtype=np.float32)

    for block in blocklist:
        y1, y2, x1, x2 = block
        print("loading data for ", block)

        rchy = 64
        rchx = 64


        pblue = scenes.nbart_blue[:, y1:y2, x1:x2].persist()
        pgreen = scenes.nbart_green[:, y1:y2, x1:x2].persist()
        pred = scenes.nbart_red[:, y1:y2, x1:x2].persist()
        pnir = scenes.nbart_nir_2[:, y1:y2, x1:x2].persist()
        pswir1 = scenes.nbart_swir_2[:, y1:y2, x1:x2].persist()
        pswir2 = scenes.nbart_swir_3[:, y1:y2, x1:x2].persist()



        chblue = pblue.chunk({"time":-1, "y":rchy, "x":rchx})
        chgreen = pgreen.chunk({"time":-1, "y":rchy, "x":rchx})
        chred = pred.chunk({"time":-1, "y":rchy, "x":rchx})
        chnir = pnir.chunk({"time":-1, "y":rchy, "x":rchx})
        chswir1 = pswir1.chunk({"time":-1, "y":rchy, "x":rchx})
        chswir2 = pswir2.chunk({"time":-1, "y":rchy, "x":rchx})

        am = gen_tsmask(chblue, chgreen, chred, chnir, chswir1, chswir2, n_ids)

        indices[y1:y2, x1:x2, :] = am.compute()
        print("Finish computing indices for ", block)
    
    
    client.close()
    
    for i, indname in enumerate(indices_list):
        
        scenes[indname] = scenes.nbart_blue[0]
        scenes[indname].data = indices[:, :, i]
    
   
    
    return scenes


def partition_blocks(irow, icol, prow, pcol):
    
    py = irow // prow + 1
    px = icol // pcol + 1
    
    blocklist = []
    for i in range(prow):
        y1=i*py
        if i == prow -1:
            y2 = irow
        else:
            y2 = (i+1)*py
            
        for j in range(pcol):
            x1 = j*px
            if j == pcol-1:
                x2 = icol
            else:
                x2 = (j+1)*px
                
            blocklist.append([y1, y2, x1, x2])
    
    return py, px, blocklist
        

    
def formfactor(mem, tn, irow, icol):
    
    gpixel = 16*1000*1000
    pnum = np.ulonglong(tn*irow *icol)
    
    mp = int(pnum / (mem*gpixel)) + 1
    
    for i in range(101):
        if (i*i>=mp):
            break
            
    if i>=10000:
        
        return -1
    
    else:
        
        return i
            

    
def prep_dask_dataset(y1, y2, x1, x2, start_of_epoch, end_of_epoch, crs, out_crs, mem):
    
    
    dc = datacube.Datacube(app='load_clearsentinel')

    s2_ds = dc.load(['s2a_ard_granule', 's2b_ard_granule'], crs = crs, output_crs=out_crs, resolution=(-20, 20), time=(start_of_epoch, end_of_epoch),
             x = (x1, x2), y = (y1, y2), group_by='solar_day', dask_chunks = {"time": 1},  measurements=['nbart_red', 'nbart_green', 'nbart_blue',
                                                             'nbart_nir_2', 'nbart_swir_2', 'nbart_swir_3'])
  
    # number of rows
    irow=s2_ds['y'].size
    # number of columns
    icol=s2_ds['x'].size
    # number of time steps
    tn = s2_ds['time'].size

    s2_ds.close()

     
    ff = formfactor(mem, tn, irow, icol)
    
    chy, chx, blocklist = partition_blocks(irow, icol, ff, ff)

    scenes = dc.load(['s2a_ard_granule', 's2b_ard_granule'], crs = crs, output_crs=out_crs, resolution=(-20, 20), time=(start_of_epoch, end_of_epoch),
             x = (x1, x2), y = (y1, y2), group_by='solar_day', dask_chunks = {"time": 1, "y": chy, "x" : chx },  
                     measurements=['nbart_red', 'nbart_green', 'nbart_blue', 'nbart_nir_2', 'nbart_swir_2', 'nbart_swir_3'])
  
    
    return scenes, blocklist