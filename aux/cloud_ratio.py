#!/usr/bin/env python
# coding: utf-8

import datacube
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl



import numpy as np

import os

sys.path.append('../lib')
import tsmask_func as tsf
import testpair as cym
import dea_datahandling as ddh



import rasterio
import xarray as xr

mpl.rcParams['figure.max_open_warning'] = 1000


# In[20]:


def rescale_rgb(rgb, percentile_stretch):
    
    lowrs, highrs = percentile_stretch
    irow = rgb.shape[0]
    icol = rgb.shape[1]
    
    for i in np.arange(3):
        onecol = rgb[:,:,i].flatten()
        ss = onecol.size
        lb = int(lowrs*ss)
        up = int(highrs*ss)
        srt = np.sort(onecol)
        vmin = srt[lb]
        vmax = srt[up]
        rg = vmax - vmin
        onecol = (onecol -vmin)/rg
        onecol[onecol<0]=0
        onecol[onecol>1]=1
        rgb[:,:,i]=onecol.reshape(irow,icol)
        
    
    return rgb
        
        


# In[21]:



def utm_code(x1, x2):
    
    mid =  (x1 + x2)/2
    
    zone = int(mid/6) + 31
    
    code = 'EPSG:327'+str(zone)
    
    return code


def std_by_paramters(data, rs, msarr):
    ntr=data.shape[1]
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=msarr[i]
        std=msarr[i+ntr]
        clm=(clm-mu)/(rs*std)
        data[:,i]=clm
        
    return data



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


 
    
    return s2_ds


# In[22]:


def readvdfile(dirc, fname):
    
    filename=dirc+'/'+fname
    dbs=xr.open_dataset(filename)
    dbs['__xarray_dataarray_variable__'].attrs['crs']=dbs['x'].crs
    dbs=dbs.rename(dict(zip(['__xarray_dataarray_variable__'], ['cloudmask'])))
    ncfilename='netcdf:'+filename
    
    upscale_factor = 1/2.0

    with rasterio.open(ncfilename) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.nearest
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    
    vdmaskval=data
    vdtsbandnames = tsf.get_timebandnames(dbs)
    dbs.close()
    
    return vdmaskval, vdtsbandnames


# In[23]:


def mapda(da, mvals):
    sp = da.shape
    da = da.flatten()
    ss = mvals.size
    mda = da.copy()
    for i in np.arange(ss):
        mda[da==i] = mvals[i]
        
    da = mda.reshape(sp)
    return da


# In[24]:


def align_two_arr(arr1, arr2):
    s1=arr1.shape
    s2=arr2.shape
    
    if (s2[0]>s1[0]):
        arr2=arr2[0:s1[0], :]
    elif (s1[0]>s2[0]):
        arr1=arr1[0:s2[0], :]
        
    
    if (s2[1]>s1[1]):
        arr2=arr2[:, 0:s1[1]]
    elif (s1[1]>s2[1]):
        arr1=arr1[:, 0:s2[1]]
        
    return arr1, arr2


# In[25]:


def mat_eval(tgpage,  vdpage):
    tgpage, vdpage = align_two_arr(tgpage, vdpage)
    mat = np.zeros((4,4), dtype=np.int32)
    maskpairs = list(zip(tgpage.flatten(), vdpage.flatten()))
    for pm in maskpairs:
        
        tv=pm[0]
        vv=pm[1]
        
 
        if (tv==0):
            tv=vv
        
        
        mat[tv, vv]+=1
    
    return mat


# In[26]:


def fpfn_rate(cmat):
    
    irow = cmat.shape[0]
   
    ss = 0
    fp = 0
    fn = 0
    
    for i in np.arange(irow):
        
        mat = cmat[i]
           

        mat = mat.reshape(4,4)
        mat = mat[1:4, 1:4]
        ss += mat.sum()
        fp += (mat[1,0] + mat[2,0])
        fn += (mat[0,1] + mat[0,2])

    ss = float(ss)
    fp = float(fp/ss)
    fn = float(fn/ss)
    
    
    return fp, fn


def main():
    
    param=sys.argv
    argc = len(param)

    if ( argc != 11 ):

        print("Usage: python3 vis_tsmask_results.py y1 y2 x1 x2 proj start_of_epoch end_of_epoch dirc loc_str vddirc vdfname")  
        print("y1: latitude of the top of the bounding box")
        print("y2: latitude of the bottom of the bounding box")
        print("x1: longitude of the left of the bounding box")
        print("x2: longitude of the right of the bounding box")
        print("proj: projection 0: EPSG:3577, 1: EPSG:4326")
        print("start_of_epoch: Start of time epoch")
        print("end_of_epoch: End of time epoch")
        print("maskdirc: directory where tsmask files located")
        print("outdirc: directory for the output image")
        print("loc_str: location string for the output file name") 
        exit()


    # latitude of the top of the bounding box
    y1 = float(param[1])

    # latitude of the bottom of the bounding box
    y2 = float(param[2])

    # longitude of the left of the bounding box
    x1 = float(param[3])

    # longitude of the right of the bounding box
    x2 = float(param[4])

    # projection 0: EPSG:3577, 1: EPSG:4326,  
    proj = int(param[5])

    # Start of time epoch
    start_of_epoch = param[6]

    # End of time epoch
    end_of_epoch = param[7]

    # directory where tsmask files located
    maskdirc = param[8]

    # directory for the output image
    outdirc = param[9]

    # location string in the output filename
    loc_str = param[10]
    
    comm ='mkdir -p '+ outdirc 
    os.system(comm)

  

    dc = datacube.Datacube(app='load_clearsentinel')


    s2_ds=load_s2_nbart_ts_cor_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
            "time": 1,
        }, proj   )

    irow=s2_ds['y'].size
    icol=s2_ds['x'].size
    tn = s2_ds['time'].size


    tsbandnames = tsf.get_timebandnames(s2_ds)
    
    print(tn, irow, icol)

    # Create numpy array to store TSmask results
    
    nmask_count = np.zeros((irow, icol), dtype=np.float32)
    fmask_count = np.zeros((irow, icol), dtype=np.float32)
    pcount = np.zeros((irow, icol), dtype=np.float32)

    

    cc = 0

    for i, tbname in enumerate(tsbandnames):

        fname = maskdirc + '/'+loc_str+'_tsmask_'+tbname+'.tif'
        if os.path.isfile(fname):
          
            fname = maskdirc + '/'+loc_str+'_tsmask_'+tbname+'.tif'
            dbs=rasterio.open(fname)
            mm = dbs.read()
            onemask = mm[0]
            dbs.close()



            fmask = s2_ds['fmask'][i].values
            fmask[fmask>3] = 1

            for y in np.arange(irow):
                for x in np.arange(icol):
                    if fmask[y,x]!=0 and onemask[y,x]!=0:
                        pcount[y,x] +=1
                        if fmask[y,x]==2 or fmask[y,x]==3:
                            fmask_count[y,x] += 1
                        if onemask[y,x]==2 or onemask[y,x]==3:
                            nmask_count[y,x] += 1
                            
                            
        print(tbname)  
            
    
            
    for y in np.arange(irow):
        for x in np.arange(icol):
            if pcount[y,x]!=0:
                fmask_count[y,x] /= pcount[y,x]
                nmask_count[y,x] /= pcount[y,x]
            else:
                fmask_count[y,x] = 0
                nmask_count[y,x] = 0
                
                
                
    geotrans = s2_ds.geobox.transform.to_gdal()
    prj = s2_ds.geobox.crs.wkt
    
    s2_ds.close()
            
    fname = outdirc + '/'+ loc_str+'_fmask_cloud_ratio.tif'
    ddh.array_to_geotiff(fname, fmask_count, geotrans, prj)

    fname = outdirc + '/'+ loc_str+'_nmask_cloud_ratio.tif'
    ddh.array_to_geotiff(fname, nmask_count, geotrans, prj)
   



if __name__ == '__main__':
    main()


