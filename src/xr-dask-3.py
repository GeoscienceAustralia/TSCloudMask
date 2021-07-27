#!/usr/bin/env python
# coding: utf-8


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



import tensorflow as tf
import tensorflow.keras.models as models
from datacube.utils.cog import write_cog



def load_bgids(indices_list, dirc, indstr, loc_str, tg_ds, chy, chx):
    
    
    
    n_ids = len(indices_list)
    
    print("Start loading long term mean of indices")


    for i, indname in enumerate(indices_list):
        fname = dirc + '/'+loc_str+'_'+indname+'_'+indstr+'.tif'
        dbs=xr.open_rasterio(fname)
           
        tg_ds[indname]=tg_ds['blue'][0]
        tg_ds[indname].data = dbs.data[0]
        tg_ds[indname]=tg_ds[indname].chunk({'y':chy, 'x':chx})

        dbs.close()
    
    
    return tg_ds



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


# In[4]:


def create_ip_data(chblue, chgreen, chred, chnir, chswir1, chswir, bgids, loc_str, outdirc):
    
    
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


# In[5]:


def dumb_getipdata(blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi):
    irow = blue.shape[0]
    icol = blue.shape[1]
    ipdata = np.zeros((irow, icol, 12), dtype = np.float32 )
    ipdata[:, :, 0] = msavi
    
    return ipdata



def cal_ip_data(blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi):
    
    #tsmask = cym.tsmask(blue, green, red, nir, swir1, swir2)
    ipdata = cym.getipdata(blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi)
    #ipdata = dumb_getipdata(blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi)
    
    return ipdata

def tf_data(blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi):
    return xr.apply_ufunc(
        cal_ip_data, blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi,
        dask='parallelized',
  
        output_core_dims= [['ipdata']], 
        dask_gufunc_kwargs = {'output_sizes' : {'ipdata' : 13}},
        output_dtypes = [np.float32]

    )



def std_by_paramters(data, rs, msarr):
    ntr=data.shape[1]
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=msarr[i]
        std=msarr[i+ntr]
        clm=(clm-mu)/(rs*std)
        data[:,i]=clm
        
    return data



def main():

    dirc ='/home/jovyan/nmask_testdata/cbr_dask_run/indices'

    loc_str ='canberra'

    ncpu = 6


    y1, y2 = -32.53284301899998, -33.52310232399998
    x1, x2 = 121.934694247, 123.105109264

    crs = 'EPSG:4326'
    out_crs = 'UTM'

    start_of_epoch = '2017-01-01'
    end_of_epoch ='2020-12-31'

    if out_crs == 'UTM':

        out_crs = tsf.utm_code(x1, x2)

    #create_local_dask_cluster(spare_mem='4GB')

    outdirc = '/home/jovyan/nmask_testdata/cbr_dask_run/maskfiles'
    modeldirc ='/home/jovyan/nmask_dask/models'
    modelname ='combine_trdata_tsmask_model'

    #Load normalisation parameters for the input data
    parafilename=modeldirc+'/'+modelname+'_standarise_parameters.npy'
    norm_paras = np.load(parafilename)
    
    #Load neural network model 
    modelfilename=modeldirc+'/'+modelname
    model = models.load_model(modelfilename)

    #Start a local cluster
    client = Client(n_workers = ncpu, threads_per_worker=1, processes = True)
    client



   


    chy, chx = 500, 500

    dc = datacube.Datacube(app='load_clearsentinel')
    tg_ds=tsf.load_s2_nbart_dask(dc, y1, y2, x1, x2, start_of_epoch, end_of_epoch, {   
            "time": 1, "y": chy, "x" : chx }, crs, out_crs )


 

    indices_list=['s6m', 'mndwi', 'msavi', 'whi']
    indstr = start_of_epoch+"_"+end_of_epoch

    # loading background indices, add the indices data to the xarray data set
    
    tg_ds = load_bgids(indices_list, dirc, indstr, loc_str, tg_ds, chy, chx)

    indfile=dirc+'/'+loc_str+'_msavi_'+indstr+'.tif'
    dbs=xr.open_rasterio(indfile)

   

    # number of rows
    irow=tg_ds['y'].size
    # number of columns
    icol=tg_ds['x'].size
    # number of time steps
    tn = tg_ds['time'].size

    tbnamelist = tsf.get_timebandnames(tg_ds)

    #Classify each scene in the time series and output the cloud mask as a cog file
    
    for i in range(tn):

        #load data for one scene 
        nmask = np.zeros(irow*icol, dtype=np.uint8)
        blue = tg_ds.blue[i, :, :].persist()
        green = tg_ds.green[i, :, :].persist()
        red = tg_ds.red[i, :, :].persist()
        nir = tg_ds.nir[i, :, :].persist()
        swir1 = tg_ds.swir1[i, :, :].persist()
        swir2 = tg_ds.swir2[i, :, :].persist()
        s6m = tg_ds.s6m.persist()
        mndwi = tg_ds.mndwi.persist()
        msavi = tg_ds.msavi.persist()
        whi = tg_ds.whi.persist()

        #prepare the input data for the nerual network model, each row of the ipdata represents a pixel
        ipdata = tf_data(blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi).compute()

        # Last column of the ipdata indicate if a pixel contains invalid input values
        ipmask = ipdata[:, :, 12].data
        tfdata = ipdata[:, :, 0:12].data

        #prepare the input data for the neural network model, filtering out invalid pixels
        ipmask = ipmask.flatten().astype(np.int)
        tfdata = tfdata.reshape(irow*icol, 12)
        tfdata = tfdata[ipmask == 1]
        tfdata = std_by_paramters(tfdata, 2, norm_paras)

        tbname = tbnamelist[i]
        print("Begin classifying scene ", tbname)
        mixtures=model.predict(tfdata)
        vdmask = np. argmax(mixtures, axis = 1) + 1

        # reconstuct the cloud mask image, invalid pixels have a cloud mask value of zero
        nmask[ipmask==1] = vdmask
        nmask = nmask.reshape(irow, icol)
        
        #Apply sptail filter to the cloud mask, eliminate cloud masks with less than 2 neighbours 
        nmask = cym.spatial_filter_v2(nmask)

        #output the cloud mask as a cog file
      
        dbs.data[0] = nmask
       
        outfname = outdirc+'/'+loc_str+'_'+tbname+'_nmask-cog.tif'
        write_cog(geo_im = dbs, fname = outfname, overwrite = True)
        print("Finish writing mask file ", outfname)


    dbs.close()
        
if __name__ == '__main__':
    main()

