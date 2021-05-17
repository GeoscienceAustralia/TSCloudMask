#!/usr/bin/env python
# coding: utf-8

import datacube
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("/g/data/u46/pjt554/dea-notebooks/Scripts")
sys.path.append("/g/data/u46/pjt554/TSmaskNN/codes")

from dea_plotting import display_map
from dea_plotting import rgb
from rasterio.enums import Resampling


import numpy as np

import os

import tsmask_func as tsf
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

    if ( argc != 13 ):

        print("Usage: python3 vis_tsmask_results.py y1 y2 x1 x2 proj start_of_epoch end_of_epoch dirc loc_str vddirc vdfname")  
        print("y1: latitude of the top of the bounding box")
        print("y2: latitude of the bottom of the bounding box")
        print("x1: longitude of the left of the bounding box")
        print("x2: longitude of the right of the bounding box")
        print("proj: projection 0: EPSG:3577, 1: EPSG:4326")
        print("start_of_epoch: Start of time epoch")
        print("end_of_epoch: End of time epoch")
        print("dirc: data directory")
        print("loc_str: location string for the output file name")
        print("maskdirc: directory where tsmask files located")
        print("vddirc: validation data directory")
        print("vdfname: file name of the validation dataset")

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

    # Data directory
    dirc = param[8]

    # location string in the output filename
    loc_str = param[9]
    
    # directory where tsmask files located
    maskdirc = param[10]

    # Validation data directory
    vddirc = param[11]
    
    # file name of the validation dataset
    vdfname = param[12]
  

    # Start of time epoch
    #start_of_epoch = '2015-01-01'

    # End of time epoch
    #end_of_epoch = '2020-12-31'


    # Output directory

    #datadirc = '/g/data/u46/pjt554/TSmaskNN/data/canberra/dataforTsmaskNN'
    #maskdirc = '/g/data/u46/pjt554/TSmaskNN/data/canberra/tsmaskbyNN'
    #outdirc = '/g/data/u46/pjt554/TSmaskNN/data/canberra/comparisons'
    
    #vddirc = '/g/data/u46/pjt554/tsmask_validation_data'

    datadirc = dirc+'/dataforTsmaskNN'
    #maskdirc = dirc+'/tsmaskbyNN'
    outdirc =  maskdirc+'/comparisons'
    
    
    comm ='mkdir -p '+ outdirc 
    os.system(comm)

    # location string in the output filename
    # loc_str = 'canberra'
    # indpred ='2015-01-01_2020-12-31'


    dc = datacube.Datacube(app='load_clearsentinel')


    s2_ds=load_s2_nbart_ts_cor_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
            "time": 1,
        }, proj   )

    irow=s2_ds['y'].size
    icol=s2_ds['x'].size
    tn = s2_ds['time'].size


    timebandsfname = datadirc + '/' + loc_str + '_timebandnames.npy'  
    tsbandnames = np.load(timebandsfname)

    print(tn, irow, icol)

    # Create numpy array to store TSmask results
    tsmask = np.zeros((tn, irow, icol), dtype=np.uint8)

    onetsmask = np.zeros(irow*icol, dtype = np.uint8)


    #vdfname = 'labels_canberra.nc'
    vdmaskval, vdtsbandnames = readvdfile(vddirc, vdfname)

    mvals=np.array([1, 3, 2, 2])
    vdmaskval = mapda(vdmaskval, mvals)
    #print(vdmaskval.shape)
    #print(vdmaskval)


    # In[30]:


    rgb= np.zeros((irow, icol, 3), dtype = np.float32)
    percentile_stretch = (0.05, 0.95)

    arrvdtsnames = np.array(vdtsbandnames)

    vtn = arrvdtsnames.size

    tsmcmat = np.zeros((vtn,16), dtype=np.int32)
    fmcmat = np.zeros((vtn,16), dtype=np.int32)


    cc = 0

    for i, tbname in enumerate(tsbandnames):

        vtidx = np.where(arrvdtsnames == tbname)[0]
        if vtidx.size > 0:
            k = vtidx[0]
            fig , [[ax1, ax2],[ax3, ax4]] = plt.subplots(2,2, constrained_layout=True)
            fig.set_size_inches(10, 10)
            fname = maskdirc + '/'+loc_str+'_tsmask_'+tbname+'.tif'
            dbs=rasterio.open(fname)
            mm = dbs.read()
            onemask = mm[0]
            dbs.close()



            red = s2_ds['red'][i].values
            green = s2_ds['green'][i].values
            blue = s2_ds['blue'][i].values
            fmask = s2_ds['fmask'][i].values
            vdmask = vdmaskval[k]

            fmask[fmask>3] = 1


            rgb[:,:, 0] = red
            rgb[:,:, 1] = green
            rgb[:,:, 2] = blue

            rgb = rescale_rgb(rgb, percentile_stretch)

            ax1.imshow(rgb)
            ax1.set_title("Real colour image")
            ax2.imshow(vdmask, vmin=0, vmax=3)
            ax2.set_title("Validation")
            ax3.imshow(fmask, vmin=0, vmax=3)
            ax3.set_title("Fmask")
            ax4.imshow(onemask, vmin=0, vmax=3)
            ax4.set_title("TSmaskML")



            fig.suptitle(loc_str+'_'+tbname)
            fname = outdirc + '/'+loc_str+'_'+tbname+'_cps.png'
            fig.savefig(fname)

            print(fname," saved")

            tsmcmat[cc, :] = mat_eval(onemask,  vdmask).flatten()
            fmcmat[cc, :] = mat_eval(fmask,  vdmask).flatten()

            cc += 1


    # In[31]:


    #s2_ds.tsmask.plot(col="time")


    # In[32]:


    #rgb(s2_ds, bands=["red", "green", "blue"], index=list(np.arange(tn)))


    # In[33]:


    fname = outdirc + '/' +loc_str+"_tsmask_vs_validation"
    np.save(fname, tsmcmat.flatten())
    fname = outdirc + '/' +loc_str+"_fmask_vs_validation"
    np.save(fname, fmcmat.flatten())
    fname = outdirc + '/' +loc_str+"_validation_ts_bandnames"
    np.save(fname, arrvdtsnames)


    # In[42]:


    cmp = np.zeros(4, dtype=np.float32)
    fp, fn = fpfn_rate(tsmcmat)
    cmp[0:2] = fp, fn
    print('fp, fn of TSmask = ', fp, fn)
    fp, fn = fpfn_rate(fmcmat)
    cmp[2:4] = fp, fn
    print('fp, fn of Fmask = ', fp, fn)

    print(cmp)
    fname = outdirc + '/' +loc_str + '_fp_fn_tsmask_fmask.npy'
    np.save(fname, cmp)
    fname = outdirc + '/' +loc_str + '_fp_fn_tsmask_fmask.txt'
    np.savetxt(fname, cmp, delimiter = ',')


    # In[43]:


    #fname = outdirc + '/' +loc_str + '_fp_fn_tsmask_fmask.npy'
    #bmp = np.load(fname)
    #print(bmp)


    # In[44]:


    #fname = outdirc + '/' +loc_str + '_fp_fn_tsmask_fmask.txt'
    #bmp = np.loadtxt(fname)
    #print(bmp)


    # In[36]:


    #tsbandnames


    # In[ ]:





    # In[ ]:

if __name__ == '__main__':
    main()


