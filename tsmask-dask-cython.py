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


# In[2]:


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
        newquery = {
        "x": (lon_left, lon_right),
        "y": (lat_top, lat_bottom),
        "time": (start_of_epoch, end_of_epoch),
        "output_crs": "EPSG:3577",
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

    # Add tsmask dataarray to the dataset

    s2_ds["tsmask"] = s2_ds["blue"]
    s2_ds["tsmask"].values = np.zeros(
        (s2_ds["time"].size, s2_ds["y"].size, s2_ds["x"].size), dtype=np.uint8)
        
    
    return s2_ds





def tsmask_filter_block(s2_ds, block, ncpu):
    
    # Define the spatial subset
    
    [r1, r2, c1, c2] = block
    print("Loading S2 NBART time series surface reflectance data for block(", r1, r2, c1, c2, ")")
    
    # scale factor
    scale = 10000.0

    # invalid value
    ivd = -999
    
    
    # Load NABR-T data from Dask array
    
   
    blue = s2_ds["blue"][:, r1:r2, c1:c2].values
    print("Finish loading the blue band")
    green = s2_ds["green"][:, r1:r2, c1:c2].values
    print("Finish loading the green band")
    red = s2_ds["red"][:, r1:r2, c1:c2].values
    print("Finish loading the red band")
    nir = s2_ds["nir"][:, r1:r2, c1:c2].values
    print("Finish loading the nir band")
    swir1 = s2_ds["swir1"][:, r1:r2, c1:c2].values
    print("Finish loading the swir1 band")
    swir2 = s2_ds["swir2"][:, r1:r2, c1:c2].values
    print("Finish loading the swir2 band")
    tsmask = s2_ds["tsmask"][:, r1:r2, c1:c2].values
    print("Finish loading the tsmask band")

    
    # initialise tsmask, all as clear pixels
    tsmask[:] = 1


 
    # detect pixels with invalid data value
    tsmask[blue<=ivd]=0
    tsmask[green<=ivd]=0
    tsmask[red<=ivd]=0
    tsmask[nir<=ivd]=0
    tsmask[swir1<=ivd]=0
    tsmask[swir2<=ivd]=0
                    
    # detect pixels with invalid data value  
    tsmask[blue == 0] = 0
    tsmask[green == 0] = 0
    tsmask[red == 0] = 0
    tsmask[nir == 0] = 0
    tsmask[swir1 == 0] = 0
    tsmask[swir2 == 0] = 0

    print("Start indices calculation")
        
    #Band average, a proxy index of brightness
    sa = ((blue+green+red+nir+swir1+swir2)/scale/6).astype(np.float32)
   
    
    # modified normalised difference water index
    mndwi = ((green - swir1) / (green + swir1)).astype(np.float32)
  
    
    # modified soil adjusted vegetation index
    msavi = ((2 * nir/scale + 1 - np.sqrt((2 * nir/scale + 1) * (2 * nir/scale + 1) - 8 * (nir/scale - red/scale))) / 2).astype(np.float32)
    


    # Band different ratio between red and blue
    wbi = ((red - blue) / blue).astype(np.float32)
   
    
    # Sum of red and blue band
    rgm = ((red + blue)/scale).astype(np.float32)
    
    
    
    # Band different ratio between green and mean of red and blue
    grbm = (green - (red + blue) / 2) / ((red + blue) / 2).astype(np.float32)
      
    
    # Bright cloud theshold
    maxcldthd = 0.45

    # label all ultra-bright pixels as clouds
    tsmask[sa > maxcldthd] = 2
    
    
    
    # Prepare tuples as input of multiprocessing 
    ts_tuples=create_ts_tuples_direct(sa, mndwi, msavi, wbi, rgm, grbm, tsmask)
    
    results = []
    
    # number of process for the  pool object
    number_of_workers = ncpu
    
    # Create a Pool object with a number of processes
    p = Pool(number_of_workers)
    
    print("Runing time series cloud and shadow detection for block(", r1, r2, c1, c2, ")")
    # Start runing the cloud detection function using a pool of independent processes
    results = p.starmap(perpixel_filter_direct_v2, ts_tuples)    
    
    # Finish the parallel runs
    p.close()
    
    # Join the results and put them back in the correct order
    p.join()

 
   # number of rows
    irow = tsmask.shape[1]

    # number of columns
    icol = tsmask.shape[2]
    
    #number of time slice
    
    tn = s2_ds["time"].size
 
    #for y in np.arange(irow):
    #    for x in np.arange(icol):
    #        s2_ds["tsmask"][:, r1+y, c1+x].values = results[y * icol + x]

    s2_ds["tsmask"][:, r1:r2, c1:c2].values = np.array(results).transpose().reshape(tn, irow, icol).copy()
    

     
    del ts_tuples
    del results
    del sa
    del mndwi 
    del msavi 
    del wbi 
    del rgm 
    del grbm 
    
    

    
    return 
    
   
    
    


# In[6]:


def create_ts_tuples_direct(sa, mndwi, msavi, wbi, rgm, grbm, tsmask):

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
    irow = tsmask.shape[1]

    # number of columns
    icol = tsmask.shape[2]

    # total number of pixels
    pnum = irow * icol

    ts_tuples = []

    for i in np.arange(pnum):

        y = int(i / icol)
        x = i % icol

        # copy time series spectral data from the data set, scale the data to float32, in range (0, 1.0)

        ts_tuples.append((sa[:, y, x], mndwi[:, y, x], msavi[:, y, x], wbi[:, y, x], rgm[:, y, x], grbm[:, y, x], tsmask[:, y, x]))

    return ts_tuples


# In[7]:


def perpixel_filter_direct_v2(sa, mndwi, msavi, wbi, rgm, grbm, tsmask):

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

    # detect single cloud / shadow pixels
    cym.testpair(sa, mndwi, 1, tsmask)
    cym.testpair(sa, mndwi, 1, tsmask)
    cym.testpair(sa, mndwi, 1, tsmask)

    # detect 2 consecutive cloud / shadow pixels
    cym.testpair(sa, mndwi, 2, tsmask)
    cym.testpair(sa, mndwi, 2, tsmask)

    # detect 3 consecutive cloud / shadow pixels
    cym.testpair(sa, mndwi, 3, tsmask)

    # detect single cloud / shadow pixels
    cym.testpair(sa, mndwi, 1, tsmask)

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



# This functioin divides the dataset into a list of blocks with smaller spatial dimensions 

def create_blocks(irow, icol, my, mx):
    
    zy=int(irow/my)+1
    zx=int(icol/mx)+1
    
    
    blist=[]
    
    for i in np.arange(zy):
        r1=i*my
        r2=r1+my
        if r2>irow:
            r2=irow
        for j in np.arange(zx):
            oneblock=np.zeros(4, dtype=int)
            c1=j*mx
            c2=c1+mx
            if (c2>icol):
                c2=icol
            
            blist.append(np.array([r1, r2, c1, c2]))
            
    return blist

def create_blocks_v2(irow, icol, ss):
    
   
    nrow=ss//icol + 1
    
    zy=irow//nrow + 1
    
    nrow = irow // zy + 1
    
    blist=[]

    
    for i in np.arange(zy):
        r1=i*nrow
        r2=r1+nrow
        if r2>irow:
            r2=irow
       
        blist.append(np.array([r1, r2, 0, icol]))
            
    return blist


param=sys.argv
argc = len(param)

if (argc != 12):
    print("Usage: python3 tsmask-dask-cython.py ncpu mem y1 y2 x1 x2 proj start_of_epoch end_of_epoch dirc, loc_str")  
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
    print("loc_str: location string in the output filename")
   
    return
    
    
   

# number of cpu cores available
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



#start_of_epoch, end_of_epoch = "2015-01-01", "2020-12-31"
#simpson desert
#(y1, y2, x1, x2) =  (-2682355.0, -2725625.0, 438225.0, 469355.0)

#Tile 55HGU
#(y1, y2, x1, x2) =  (-36.99497079, -38.01366742, 149.24791461, 150.52675786)


#Tile 55JGK
#(y1, y2, x1, x2) = (-27.087734,  -28.09861708, 149.01710236, 150.15181468)
#(y1, y2, x1, x2) = (-27.087734,  -27.59315, 149.01710236, 149.5844)


dc = datacube.Datacube(app='load_clearsentinel')

tg_ds=load_s2_nbart_ts_cor_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
        "time": 1,
    }, proj   )




irow=tg_ds['y'].size
icol=tg_ds['x'].size
tn = tg_ds['time'].size

#mem=128

print(irow, icol, tn)

ss = int(720000*2.1*550*mem/tn/64)

blist=create_blocks_v2(irow, icol, ss)

for block in blist:
      
    tsmask_filter_block(tg_ds, block, ncpu)




results = []

# number of process for the  pool object
number_of_workers = ncpu

# Create a Pool object with a number of processes
p = Pool(number_of_workers)

# create a list of scene
paralist = [tg_ds["tsmask"].values[i, :, :] for i in np.arange(tg_ds.time.size)]

# Start runing the spatial filter function using a pool of indepedent processes
results = p.map(cym.spatial_filter, paralist)


# Finish the parallel runs
p.close()

# Join the results and put them back in the correct order
p.join()


# Save the cloud/shadow masks to the 'tsmask' dataarray in the s2_ds dataset
for i in np.arange(tg_ds.time.size):
    tg_ds["tsmask"].values[i, :, :] = results[i]
    

#output the tsmask to as cloud optimised geotiff files
bandsets=['tsmask']
outbandnames=['tsmask']
#dirc='/g/data/u46/pjt554/tsmask_validation_data/simpson'
#loc_str='simpson'
#dirc='/g/data/u46/pjt554/tsmask_validation_data/55JGK'
#loc_str='55JGK'

tsf.output_ds_to_cog(bandsets, outbandnames, dirc, loc_str, tg_ds)



# check if the output file is correct
#testfile=dirc+'/simpson_tsmask_2017-10-10.tif'
#onemask=rasterio.open(testfile)
#aa = onemask.read()
#print(onemask.bounds)
#print(onemask.crs)
#onemask.transform*(0,0)





print(tg_ds['tsmask'][:,53, 3])

print(tg_ds['tsmask'][:, 53, 53])







