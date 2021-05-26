
import rasterio
import xarray as xr
import sys
import datacube
import numpy as np

from multiprocessing import Pool
import dask.array as da
import os
import dask
from dask.distributed import Client

sys.path.append('../lib')
import tsmask_func as tsf
import testpair as cym
import dea_datahandling as ddh


def prep_features(sa, mndwi, msavi, whi, bg_indices, ipdata):
    n_ids=4
    for k in range(n_ids):
        ipdata[: , k*3] = bg_indices[k, :]
        
    ipdata[:, 1] = sa
    ipdata[:, 4] = mndwi
    ipdata[:, 7] = msavi
    ipdata[:, 10] = whi
    
    for k in range(n_ids):
        ipdata[: , k*3+2] = (ipdata[: , k*3+1] - ipdata[: , k*3]) / ipdata[: , k*3]
        
    
    return ipdata    


   
def tsmask_filter_onearea(s2_ds, ncpu, tsmask):
   
  
    
    # Load NABR-T data from Dask array
    print(s2_ds)
   
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

 
  
    # Number of time slice
    
    tn = s2_ds["time"].size

    # Copy cloud mask results to tsmask    
    tsmask = np.array(results).transpose().reshape(tn, irow, icol).copy()
  
   
    
    del ts_tuples
    del results
 
    

    
    return tsmask
    

# Prepare tuples as input of multiprocessing 

def create_ts_tuples_direct_tsmask(blue, green, red, nir, swir1, swir2, tsmask, irow ,icol):

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

        ts_tuples.append((blue[:, y, x], green[:, y, x], red[:, y, x], nir[:, y, x], swir1[:, y, x], 
                          swir2[:, y, x], tsmask[:, y, x]))

    return ts_tuples



# Prepare tuples as input of multiprocessing 

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


def bg_indices_onearea(s2_ds, ncpu, tsmask):
   
  
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
    irow=s2_ds['y'].size
    # number of columns
    icol=s2_ds['x'].size
    # number of time steps
    tn = s2_ds['time'].size

 
    # Prepare tuples as input of multiprocessing 
    ts_tuples=create_ts_tuples_direct_tsmask(blue, green, red, nir, swir1, swir2, tsmask, irow, icol)
    
    results = []
    
    # number of process for the  pool object
    number_of_workers = ncpu
    
    # Create a Pool object with a number of processes
    p = Pool(number_of_workers)
    
    print("Begin calculating long term mean of indices for block(", 0, irow, 0, icol, ")")
    
    # Start runing the cloud detection function using a pool of independent processes
    results = p.starmap(cym.perpixel_bg_indices_core, ts_tuples)   
   
    p.close()
    
    # Join the results and put them back in the correct order
    p.join()
    
    print("Finish calculating long term mean of indices for block(", 0, irow, 0, icol, ")")

 
    #number of output indices
    
    n_ids = 5
   
    bgdis = np.zeros((n_ids, irow, icol), dtype=np.float32)

    bgids = np.array(results).transpose().reshape(n_ids, irow, icol).copy()
  
   
    
    del ts_tuples
    del results
 
    

    
    return bgids
    
       
    


def bg_indices_one_iteration(ncpu, tg_ds, dirc, loc_str, tsmask, start_of_epoch, end_of_epoch):
    
     
   
    irow=tg_ds['y'].size
    icol=tg_ds['x'].size
    tn = tg_ds['time'].size


    print(tn , irow, icol)
    
    print("Calculting long term means of SR derived indices for area")
    
    
    
    # Run time series cloud mask algorithm on the data 
    bgids = bg_indices_onearea(tg_ds, ncpu, tsmask)
    
    
    geotrans = tg_ds.geobox.transform.to_gdal()
    prj = tg_ds.geobox.crs.wkt
    
    indices_list=['msavi', 'mndwi', 's6m', 'whi']
    
    print("Begin writing long term mean of indices files")

    for i, indname in enumerate(indices_list):
        fname = dirc + '/'+loc_str+'_'+indname+'_'+start_of_epoch+'_'+end_of_epoch+'.tif'
        ddh.array_to_geotiff(fname, bgids[i], geotrans, prj)
   

    bgids = bgids[0:4]
    bgids = bgids.reshape(4, irow*icol)
    return bgids
    
    
def create_ip_data(s2_ds, bgids, loc_str, outdirc):
    
    
    irow=s2_ds['y'].size
    icol=s2_ds['x'].size
    tn = s2_ds['time'].size

    timebandnames = tsf.get_timebandnames(s2_ds)
    
    pnum = irow * icol
    
    n_ids = bgids.shape[0]
    
    ipdata=np.zeros((pnum, n_ids*3), dtype=np.float32)

    for i in np.arange(tn):

        blue = s2_ds["blue"][i].values
        #print("Finish loading the blue band")  
    
        green = s2_ds["green"][i].values
        #print("Finish loading the green band")

        red = s2_ds["red"][i].values
        #print("Finish loading the red band")

        nir = s2_ds["nir"][i].values
        #print("Finish loading the nir band")

        swir1 = s2_ds["swir1"][i].values
        #print("Finish loading the swir1 band")

        swir2 = s2_ds["swir2"][i].values
        #print("Finish loading the swir2 band")
    
   
  
    
    
        #convert cal_indices in cython
        sa, mndwi, msavi, whi, mask = cym.cal_indices(blue.flatten(), green.flatten(), red.flatten(), nir.flatten(), swir1.flatten(), swir2.flatten(), pnum)

        print(timebandnames[i])
        ipdata = prep_features(sa, mndwi, msavi, whi, bgids, ipdata)
        #print(ipdata[:, 11])
        #print(i, mask.shape, mask.sum())
        vdipdata = ipdata[mask==1]
        #print(vdipdata.shape, vdipdata[:,11])
        vdipdata = vdipdata.flatten()


        datafname = outdirc + '/' + loc_str + '_'+timebandnames[i]+'_ipdata'
        np.save(datafname, vdipdata)
        maskfname = outdirc + '/' + loc_str + '_'+timebandnames[i]+'_ipmask'
        np.save(maskfname, mask)
        tsbandfname = outdirc + '/' + loc_str + '_timebandnames'
        np.save(tsbandfname, timebandnames)


def tsmask_one_iteration(ncpu, mem, block, crs, out_crs, start_of_epoch, end_of_epoch, dirc, loc_str):
    
    
    
    [y1, y2, x1, x2] = block
    
    #Datacube object
    
    dc = datacube.Datacube(app='load_clearsentinel')

    
    tg_ds=tsf.load_s2_nbart_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
            "time": 1,
        }, crs, out_crs )


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
    
    
    
    print("Begin calculting long term of the indice set")
    bgids = bg_indices_one_iteration(ncpu, tg_ds, dirc, loc_str, tsmask, start_of_epoch, end_of_epoch)
    
    print(bgids.shape)
    
   # print("Begin creating input features for Nmask ANN model")
   # create_ip_data(tg_ds, bgids, loc_str, dirc)
    
    
       
    tg_ds.close()


# This functioin divides the dataset into a list of blocks with smaller spatial dimensions 

def create_blocks_v4(irow, icol, tn, y1, y2, x1, x2, mem):
    
    # Memory factor, the higher value, the large memory footprint of the filter algorithm 
    mfr = 4.6

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
    
    
def utm_code(x1, x2):
    
    mid =  (x1 + x2)/2
    
    zone = int(mid/6) + 31
    
    code = 'EPSG:327'+str(zone)
    
    return code


    
    
def main():
    
    param=sys.argv
    argc = len(param)

    if ( argc != 13 ):

        print("Usage: python3 nmask_p0.py ncpu mem y1 y2 x1 x2 crs out_crs start_of_epoch end_of_epoch dirc loc_str")  
        print("ncpu: number of cpu cores available")
        print("mem: system memory in GB")
        print("y1: latitude of the top of the bounding box")
        print("y2: latitude of the bottom of the bounding box")
        print("x1: longitude of the left of the bounding box")
        print("x2: longitude of the right of the bounding box")
        print("crs: projection string")
        print("out_crs: output projection string")
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

    # projection string, such as EPSG:3577, 1: EPSG:4326,  
    crs = param[7]

    # output projection string: such as EPSG:3577, 1: EPSG:4326,  
    out_crs = param[8]
    
    
    if out_crs == 'UTM':
        
        out_crs = utm_code(x1, x2)
    
    
    # Start of time epoch
    start_of_epoch = param[9]

    # End of time epoch
    end_of_epoch = param[10]

    # Output directory
    dirc = param[11]

    # location string in the output filename
    loc_str = param[12]




    comm ='mkdir -p '+ dirc 
    os.system(comm)

    #dc = datacube.Datacube(app='load_clearsentinel')

    #tg_ds=tsf.load_s2_nbart_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
    #        "time": 1,
    #    }, crs, out_crs   )


    #irow=tg_ds['y'].size
    #icol=tg_ds['x'].size
    #tn = tg_ds['time'].size

    
    #print(tn , irow, icol)

   

    # Divide the dataset in multiple smaller chunks 
    #blist=create_blocks_v4(irow, icol, tn, y1, y2, x1, x2, mem)

    blist =[]
    blist.append([y1, y2, x1, x2])
    
    # Run time series cloud detection function in chunks
    
    cc = 1
    ss = len(blist)
    
    if (ss==0):
        exit()
    
    for block in blist:

        if ss>1:
            cur_loc_str = loc_str+'-'+ str(cc) +'-of-' +str(ss) 
        else:
            cur_loc_str = loc_str
        
        tsmask_one_iteration(ncpu, mem, block, crs, out_crs, start_of_epoch, end_of_epoch, dirc, cur_loc_str)
        cc += 1

    
    #tg_ds.close()
    


if __name__ == '__main__':
    main()
