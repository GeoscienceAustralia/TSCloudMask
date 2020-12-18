#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rasterio
import xarray as xr
import sys
import datacube
import dea_datahandling as ddh
import numpy as np
import tsmask_func as tsf
from multiprocessing import Pool
import dask.array as da
import testpair as cym


import dask
from dask.distributed import Client



# In[2]:
def utm_code(x1, x2):
    
    mid =  (x1 + x2)/2
    
    zone = int(mid/6) + 31
    
    code = 'EPSG:327'+str(zone)
    
    return code

def load_s2_nbart_ts_cor_dask(
    dc, lat_top, lat_bottom, lon_left, lon_right, start_of_epoch, end_of_epoch, chunks, cor_type
):

    
    allbands = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3",
        "fmask",
    ]
    
    # Define spatial and temporal coverage

    if (cor_type==0):
        newquery = {
        "crs": "EPSG:3577",
        "x": (lon_left, lon_right),
        "y": (lat_top, lat_bottom),
        "time": (start_of_epoch, end_of_epoch),
        "output_crs": "EPSG:3577",
        "resolution": (-20, 20),
        "measurements": allbands,
        "dask_chunks": chunks,
        "group_by": "solar_day",
        }
    elif (cor_type==1):
        
        #UTM projection zone code
        outcrs = utm_code(lon_left, lon_right)
        
        newquery = {
        "x": (lon_left, lon_right),
        "y": (lat_top, lat_bottom),
        "time": (start_of_epoch, end_of_epoch),
        "output_crs": outcrs,
        "resolution": (-20, 20),
        "measurements": allbands,
        "dask_chunks": chunks,
        "group_by": "solar_day",
        }

    # Names of targeted spectral bands


    # Band names used with in the dataset
    new_bandlabels = ["blue", "green", "red", "nir", "swir1", "swir2", "fmask"]

    # Load S2 data using Datacube API
 

    s2_ds = ddh.load_ard(dc,
             products=['s2a_ard_granule', 's2b_ard_granule'],
             min_gooddata=0.0,
             mask_pixel_quality=False,
             **newquery)
    
    # Rename spectral band names to new band labels
    rndic = dict(zip(allbands, new_bandlabels))
    s2_ds = s2_ds.rename(rndic)


  #  s2_ds["tsmask"] = s2_ds["blue"]
  #  s2_ds["tsmask"].values = np.zeros(
  #      (s2_ds["time"].size, s2_ds["y"].size, s2_ds["x"].size), dtype=np.uint8)
        
    
    return s2_ds


def tsmask_filter_block(s2_ds, block, ncpu, mtsmask):
    
    # Define the spatial subset
    
    [r1, r2, c1, c2] = block
    print("Loading S2 NBART time series surface reflectance data for block(", r1, r2, c1, c2, ")")
    
  
    
    # Load NABR-T data from Dask array
    
   
    #blue = s2_ds["blue"][:, r1:r2, c1:c2].values
    blue = s2_ds["blue"].values
    print("Finish loading the blue band")
    #green = s2_ds["green"][:, r1:r2, c1:c2].values
    green = s2_ds["green"].values
    print("Finish loading the green band")
    #red = s2_ds["red"][:, r1:r2, c1:c2].values
    red = s2_ds["red"].values
    print("Finish loading the red band")
    #nir = s2_ds["nir"][:, r1:r2, c1:c2].values
    nir = s2_ds["nir"].values
    print("Finish loading the nir band")
    #swir1 = s2_ds["swir1"][:, r1:r2, c1:c2].values
    swir1 = s2_ds["swir1"].values
    print("Finish loading the swir1 band")
    #swir2 = s2_ds["swir2"][:, r1:r2, c1:c2].values
    swir2 = s2_ds["swir2"].values
    print("Finish loading the swir2 band")
    
    
    tsmask = mtsmask[:, r1:r2, c1:c2]

    # number of rows
    irow = tsmask.shape[1]

    # number of columns
    icol = tsmask.shape[2]
    


    # Prepare tuples as input of multiprocessing 
    ts_tuples=create_ts_tuples_direct(blue, green, red, nir, swir1, swir2, irow, icol)
    
    results = []
    
    # number of process for the  pool object
    number_of_workers = ncpu
    
    # Create a Pool object with a number of processes
    p = Pool(number_of_workers)
    
    print("Begin runing time series cloud and shadow detection for block(", r1, r2, c1, c2, ")")
    
    # Start runing the cloud detection function using a pool of independent processes
    results = p.starmap(cym.perpixel_filter_direct_core, ts_tuples)   
   
    p.close()
    
    # Join the results and put them back in the correct order
    p.join()
    
    print("Finish time series cloud and shadow detection for block(", r1, r2, c1, c2, ")")

 
  
    #number of time slice
    
    tn = s2_ds["time"].size

    mtsmask[:, r1:r2, c1:c2] = np.array(results).transpose().reshape(tn, irow, icol).copy()
  
     
    del ts_tuples
    del results
 
    

    
    return 
    

    
def tsmask_filter_onearea(s2_ds, ncpu, tsmask):
   
  
    
    # Load NABR-T data from Dask array
    
   
    blue = s2_ds["blue"].values
    print("Finish loading the blue band")
   
    green = s2_ds["green"].values
    print("Finish loading the green band")
   
    red = s2_ds["red"].values
    print("Finish loading the red band")
   
    nir = s2_ds["nir"].values
    print("Finish loading the nir band")
   
    swir1 = s2_ds["swir1"].values
    print("Finish loading the swir1 band")
   
    swir2 = s2_ds["swir2"].values
    print("Finish loading the swir2 band")
    
    
   
    # number of rows
    irow = tsmask.shape[1]

    # number of columns
    icol = tsmask.shape[2]
    


    # Prepare tuples as input of multiprocessing 
    ts_tuples=create_ts_tuples_direct(blue, green, red, nir, swir1, swir2, irow, icol)
    
    results = []
    
    # number of process for the  pool object
    number_of_workers = ncpu
    
    # Create a Pool object with a number of processes
    p = Pool(number_of_workers)
    
    print("Begin runing time series cloud and shadow detection for block(", 0, irow, 0, icol, ")")
    
    # Start runing the cloud detection function using a pool of independent processes
    results = p.starmap(cym.perpixel_filter_direct_core, ts_tuples)   
   
    p.close()
    
    # Join the results and put them back in the correct order
    p.join()
    
    print("Finish time series cloud and shadow detection for block(", 0, irow, 0, icol, ")")

 
  
    #number of time slice
    
    tn = s2_ds["time"].size

    tsmask = np.array(results).transpose().reshape(tn, irow, icol).copy()
  
   
    
    del ts_tuples
    del results
 
    

    
    return tsmask
    
       
    



def create_ts_tuples_direct(blue, green, red, nir, swir1, swir2, irow ,icol):

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
   
     
    # total number of pixels
    pnum = irow * icol

    ts_tuples = []

    for i in np.arange(pnum):

        y = int(i / icol)
        x = i % icol

        # copy time series spectral data from the data set, scale the data to float32, in range (0, 1.0)

        ts_tuples.append((blue[:, y, x], green[:, y, x], red[:, y, x], nir[:, y, x], swir1[:, y, x], swir2[:, y, x]))

    return ts_tuples





# This functioin divides the dataset into a list of blocks with smaller spatial dimensions 


def create_blocks_v2(irow, icol, ss):
    
    

    nrow=ss//icol + 1
    
    zy=irow//nrow + 1
  
    
    blist=[]

    
    for i in np.arange(zy):
        r1=i*nrow
        r2=r1+nrow
        if r2>irow:
            r2=irow
       
        blist.append(np.array([r1, r2, 0, icol]))
            
    return blist


def tsmask_one_iteration(ncpu, mem, block, proj, start_of_epoch, end_of_epoch, dirc, loc_str):
    
    
    
    [y1, y2, x1, x2] = block
    #Datacube object
    
    dc = datacube.Datacube(app='load_clearsentinel')

    
    tg_ds=load_s2_nbart_ts_cor_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
            "time": 1,
        }, proj   )


    memstr=str(mem)+'GB'
    
    client=Client(n_workers=ncpu, threads_per_worker=2, memory_limit = memstr)
    
    client.compute(tg_ds)

    client.close()
    
    irow=tg_ds['y'].size
    icol=tg_ds['x'].size
    tn = tg_ds['time'].size


    print(tn , irow, icol)
    
    # Create numpy array to store TSmask results
    tsmask = np.zeros((tn, irow, icol), dtype=np.uint8)

    print("Time series cloud and shadow detection for area (", y1, y2, x1, x2, ")")
    
    # Run time series cloud mask algorithm on the data 
    tsmask = tsmask_filter_onearea(tg_ds, ncpu, tsmask)
    
    
      
     
    print("Begin applying spatial filter")

    results = []

    # number of process for the  pool object
    number_of_workers = ncpu

    # Create a Pool object with a number of processes
    p = Pool(number_of_workers)

    # create a list of scene
    paralist = [tsmask[i, :, :] for i in range(tn)]

    # Start runing the spatial filter function using a pool of indepedent processes
    results = p.map(cym.spatial_filter_v2, paralist)


    # Finish the parallel runs
    p.close()

    # Join the results and put them back in the correct order
    p.join()


    # Save the cloud/shadow masks to the 'tsmask' dataarray in the s2_ds dataset
    for i in range(tn):
        tsmask[i, :, :] = results[i]
    
    
    
    
    tg_ds["tsmask"] = tg_ds["blue"]
    tg_ds["tsmask"].values = tsmask

    print("Begin writing output files")


    #output the tsmask as cloud optimised geotiff files
    bandsets=['tsmask']
    outbandnames=['tsmask']

    tsf.output_ds_to_cog(bandsets, outbandnames, dirc, loc_str, tg_ds)

    # Some peeks into the results, for validation purpose only, will be deleted later
    
    
    print(tg_ds['tsmask'][:, 53, 3])

    print(tg_ds['tsmask'][:, 153, 853])

    print(tg_ds['tsmask'][100, :, :])
    
    tg_ds.close()



def create_blocks_v3(irow, icol, tn, y1, y2, x1, x2, mem):
    
    # Memory factor, the higher value, the large memory footprint of the filter algorithm 
    mfr = 3.6

    # Calculate how many pixels/time stamps can be accommodated by the system memory
    ss = int(720000*mfr*550*mem/tn/64)

    blist=[]
    pnum=icol*irow
    if (ss>=pnum):
        blist.append([y1, y2, x1, x2])
    else:
        pf = pnum//ss+1
        width = (x2-x1)/pf
        xl = x1
        xr = x1+width
        for i in range(pf):
            blist.append([y1, y2, xl, xr])
            xl = xr
            xr += width
            
    
   
    return blist

   

def create_blocks_v4(irow, icol, tn, y1, y2, x1, x2, mem):
    
    # Memory factor, the higher value, the large memory footprint of the filter algorithm 
    mfr = 3.6

    # Calculate how many pixels/time stamps can be accommodated by the system memory
    ss = int(720000*mfr*550*mem/tn/64)

    blist=[]
    pnum=icol*irow
    
    if (ss>=pnum):
        blist.append([y1, y2, x1, x2])
    else:
        pf = pnum//ss+1
        if pf > 4:
            print("Area too large for one node")
        else:
            mx = (x1+x2)/2
            my = (y1+y2)/2
            blist.append([y1, my, x1, mx])
            blist.append([y1, my, mx, x2])
            blist.append([my, y2, x1, mx])
            blist.append([my, y2, mx, x2])
      
   
    return blist
    
def main():
    
    param=sys.argv
    argc = len(param)

    if ( argc != 12 ):

        print("Usage: python3 tsmask-dask-cython.py ncpu mem y1 y2 x1 x2 proj start_of_epoch end_of_epoch dirc loc_str")  
        print("ncpu: number of cpu cores available")
        print("mem: system memory in GB")
        print("y1: latitude of the top of the bounding box")
        print("y2: latitude of the bottom of the bounding box")
        print("x1: longitude of the left of the bounding box")
        print("x2: longitude of the right of the bounding box")
        print("proj: projection 0: EPSG:3577, 1: EPSG:4326")
        print("start_of_epoch: Start of time epoch")
        print("end_of_epoch: End of time epoch")
        print("dirc: Output directory")
        print("loc_str: location string for the output file name")


        exit()


    ## number of cpu cores available
    ncpu = int(param[1])

    # system memory in GB
    mem = int(param[2])

    # latitude of the top of the bounding box
    y1 = float(param[3])

    # latitude of the bottom of the bounding box
    y2 = float(param[4])

    # longitude of the left of the bounding box
    x1 = float(param[5])

    # longitude of the right of the bounding box
    x2 = float(param[6])

    # projection 0: EPSG:3577, 1: EPSG:4326,  
    proj = int(param[7])

    # Start of time epoch
    start_of_epoch = param[8]

    # End of time epoch
    end_of_epoch = param[9]

    # Output directory
    dirc = param[10]

    # location string in the output filename
    loc_str = param[11]




    #Tile 55HGU
    #(y1, y2, x1, x2) =  (-36.99497079, -38.01366742, 149.24791461, 150.52675786)




    dc = datacube.Datacube(app='load_clearsentinel')

    tg_ds=load_s2_nbart_ts_cor_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
            "time": 1,
        }, proj   )


    irow=tg_ds['y'].size
    icol=tg_ds['x'].size
    tn = tg_ds['time'].size

    
    print(tn , irow, icol)

   

    # Divide the dataset in multiple smaller chunks 
    blist=create_blocks_v4(irow, icol, tn, y1, y2, x1, x2, mem)

    # Run time series cloud detection function in chunks
    
    cc = 1
    ss = len(blist)
    
    if (ss==0):
        exit()
    
    for block in blist:

        cur_loc_str = loc_str+'-'+ str(cc) +'-of-' +str(ss) 
        tsmask_one_iteration(ncpu, mem, block, proj, start_of_epoch, end_of_epoch, dirc, cur_loc_str)
        cc += 1

    
    tg_ds.close()
    


if __name__ == '__main__':
    main()
