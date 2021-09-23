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
from datacube.testutils.io import dc_read
from datacube.virtual import Transformation, Measurement
from xarray import Dataset

import tensorflow as tf
import tensorflow.keras.models as models


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
    
    
    """

    Description: 
    
    This function partitions a 2D scene into a set of smaller 2-D blocks
  
    Parameters: 
    
        Names of variables: irow, icol
        Descriptions: size of the 2-D scene
        Data types and formats: int32 
            
        Names of variables: prow, pcol
        Descriptions: number of partitions in y and x directions
        Data types and formats: int32
    
    Return:  
    
        Names of variables: py, px
        Descriptions: size of sub blocks
        Data types and formats: int32 
        
        Names of variables: blocklist
        Descriptions: a list of bounding boxes, which define a set of sub blocks
        Data types and formats: int32
        
    
        
   
    """
    
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
            


        
def load_s2_ard(y1, y2, x1, x2, start_of_epoch, end_of_epoch, crs, out_crs, chunks):
    
    
    
    allbands = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3"
    ]
    
  
    query = {
        "crs": crs,
        "x": (x1, x2),
        "y": (y1, y2),
        "time": (start_of_epoch, end_of_epoch),
        "output_crs": out_crs,
        "resolution": (-20, 20),
        "measurements": allbands,
        "group_by": "solar_day"
        }
    
    
    dc = datacube.Datacube(app='load_clearsentinel')
    dss1 = dc.find_datasets(product = 's2a_ard_granule', **query)
    dss2 = dc.find_datasets(product = 's2b_ard_granule', **query)
    
    
    newquery = {
        "crs": crs,
        "x": (x1, x2),
        "y": (y1, y2),
        "time": (start_of_epoch, end_of_epoch),
        "output_crs": out_crs,
        "resolution": (-20, 20),
        "measurements": allbands,
        "dask_chunks": chunks,
        "group_by": "solar_day"
        }
    
    
    
    im = dc.load(datasets=dss1+dss2, **newquery)
    
    return im
        

    
def prep_dask_dataset(y1, y2, x1, x2, start_of_epoch, end_of_epoch, crs, out_crs, mem):
    
 
    
    chunks ={"time": 1}
    s2_ds = load_s2_ard(y1, y2, x1, x2, start_of_epoch, end_of_epoch, crs, out_crs, chunks)
    
    
    # number of rows
    irow=s2_ds['y'].size
    # number of columns
    icol=s2_ds['x'].size
    # number of time steps
    tn = s2_ds['time'].size

    s2_ds.close()

     
    ff = formfactor(mem, tn, irow, icol)
    
    chy, chx, blocklist = partition_blocks(irow, icol, ff, ff)

    chunks = {"time": 1, "y": chy, "x" : chx }
    
    scenes = load_s2_ard(y1, y2, x1, x2, start_of_epoch, end_of_epoch, crs, out_crs, chunks)
  
    return scenes, blocklist


def std_by_paramters(data, rs, msarr):
    
    ntr=data.shape[1]
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=msarr[i]
        std=msarr[i+ntr]
        clm=(clm-mu)/(rs*std)
        data[:,i]=clm

    return data



def cal_ip_data_vb(blue, green, red, nir, swir1, swir2, s6m, s6m_std, mndwi, mndwi_std, msavi, msavi_std, whi, whi_std):
    
    """

    Description: 
    
    This function calculates input features for the Nmask ANN model 
  
    Parameters: 
    
        Names of variables: blue, green, red, nir, swir1, swir2
        Descriptions: S2 spectral bands
        Data types and formats: 2-D array of int16 
            
        Names of variables: s6m, s6m_std, mndwi, mndwi_std, msavi, msavi_std, whi, whi_std
        Descriptions: long term mean and standard deviations of s6m, mndwi, 
        Data types and formats: int32
    
    Return:  
    
        Names of variables: ipdata
        Descriptions: input features for the Nmask ANN model 
        Data types and formats: 2-D array of float32 
        
           
   
    """
   
    ipdata = cym.getipdata_vb(blue, green, red, nir, swir1, swir2, s6m, s6m_std, mndwi, mndwi_std, msavi, msavi_std, whi, whi_std)
       
    return ipdata




def tf_data_vb(blue, green, red, nir, swir1, swir2, s6m, s6m_std, mndwi, mndwi_std, msavi, msavi_std, whi, whi_std):
    return xr.apply_ufunc(
        cal_ip_data_vb, blue, green, red, nir, swir1, swir2, s6m, s6m_std, mndwi, mndwi_std, msavi, msavi_std, whi, whi_std,
        dask='parallelized',
        output_core_dims= [['ipdata']], 
        dask_gufunc_kwargs = {'output_sizes' : {'ipdata' : 13}},
        output_dtypes = [np.float32]
    )



def nmask_transform(scenes, medians, client):
    
    # location and name of the Nmask ANN model
    # need to be set to a relative location with in the model when it is packaged
    modeldirc ='/home/jovyan/tsmask_repos/TSCloudMask/models'
    modelname ='nmask_vb_comb'

    #Load normalisation parameters for the input data
    parafilename=modeldirc+'/'+modelname+'_standarise_parameters.npy'
    norm_paras = np.load(parafilename)

    #Load neural network model 
    modelfilename=modeldirc+'/'+modelname+'_model'
    model = models.load_model(modelfilename)

    chy, chx = 500, 500

    # number of rows
    irow=scenes.coords['y'].size
    # number of columns
    icol=scenes.coords['x'].size
    # number of time steps
    tn = scenes.coords['time'].size

    tbnamelist = get_timebandnames(scenes.coords)

    #Classify each scene in the time series and output the cloud mask as a cog file
    for i in range(tn):
        #load data for one scene 
        nmask = np.zeros(irow*icol, dtype=np.uint8)
        blue = scenes.nbart_blue[i, :, :].persist()
        green = scenes.nbart_green[i, :, :].persist()
        red = scenes.nbart_red[i, :, :].persist()
        nir = scenes.nbart_nir_2[i, :, :].persist()
        swir1 = scenes.nbart_swir_2[i, :, :].persist()
        swir2 = scenes.nbart_swir_3[i, :, :].persist()
        s6m = medians.s6m.persist()
        mndwi = medians.mndwi.persist()
        msavi = medians.msavi.persist()
        whi = medians.whi.persist()
        s6m_std = medians.s6m_std.persist()
        mndwi_std = medians.mndwi_std.persist()
        msavi_std = medians.msavi_std.persist()
        whi_std = medians.whi_std.persist()

        #prepare the input data for the nerual network model, each row of the ipdata represents a pixel
        #ipdata = tf_data(blue, green, red, nir, swir1, swir2, s6m, mndwi, msavi, whi).compute()
        
        ipdata = tf_data_vb(blue, green, red, nir, swir1, swir2, s6m, s6m_std, mndwi, mndwi_std, msavi, msavi_std, whi, whi_std).compute()
        
        # Last column of the ipdata indicate if a pixel contains invalid input values
        ipmask = ipdata[:, :, 12].data
        tfdata = ipdata[:, :, 0:12].data

        #prepare the input data for the neural network model, filtering out invalid pixels
        ipmask = ipmask.flatten().astype(np.int8)
        tfdata = tfdata.reshape(irow*icol, 12)
        tfdata = tfdata[ipmask == 1]
        tfdata = std_by_paramters(tfdata, 2, norm_paras)

        tbname = tbnamelist[i]
        print("Begin classifying scene ", tbname)
        mixtures=model.predict(tfdata)
        vdmask = np.argmax(mixtures, axis = 1) + 1

        # reconstuct the cloud mask image, invalid pixels have a cloud mask value of zero
        nmask[ipmask==1] = vdmask
        nmask = nmask.reshape(irow, icol)

        #Apply sptail filter to the cloud mask, eliminate cloud masks with less than 2 neighbours 
        nmask = cym.spatial_filter_v2(nmask)
        nmask = cym.spatial_filter_shadow(nmask)
        
        nmask = cym.spatial_filter_v2(nmask)
        nmask = cym.spatial_filter_shadow(nmask)

        #output the cloud mask as a cog file
        yield nmask
        
        
        
NMASK_OUTPUT = [{
    'name': 'nmask',
    'dtype': 'uint8',
    'nodata': 0,
    'clear': 1,
    'cloud': 2,
    'shadow': 3,
    'units': '1'
}, ]

def _to_xrds_coords(geobox):
    return {dim: coord.values for dim, coord in geobox.coordinates.items()}



class NmaskClassifier(Transformation):
    
    def __init__(self, median_path=None, indstr=None, n_workers=2):
        
        # directory or location where the long term mean of indices locate
        self.median_path = median_path
        # Patterns of indices file,.e.g '56HLH_{}_2015-01-01_2020-12-31-cog.tif' 
        self.indstr = indstr
        # number of workers of Dask client
        self.n_workers = n_workers
        
        self.output_measurements = {m['name']: Measurement(**m) for m in NMASK_OUTPUT}
        
   
    def measurements(self, input_measurements):
        return self.output_measurements

    def compute(self, data) -> Dataset:
        client = Client(n_workers = self.n_workers, threads_per_worker=1, processes = True)
        medians = self._load_medians(data.geobox)
        nmasks = []
        for nmask in nmask_transform(data, medians=medians, client=client):
            nmasks.append(nmask)
        
        nmasks = xr.Dataset({'nmask': (('time', 'y', 'x'), nmasks)}, coords=data.coords)
        nmasks.attrs['crs'] = data.attrs['crs']
        return nmasks

    def _load_medians(self, gbox):
        
        indices_list=['s6m', 's6m_std', 'mndwi', 'mndwi_std', 'msavi', 'msavi_std', 'whi', 'whi_std']
        medians = {ind: dc_read('{}/{}'.format(self.median_path, self.indstr.format(ind)), gbox=gbox, resampling="bilinear")
                   for ind in indices_list}
        return xr.Dataset(
            data_vars={ind: (('y', 'x'), medians[ind])
                        for ind in indices_list},
            coords=_to_xrds_coords(gbox),
            attrs={'crs': gbox.crs}
        )
    
    
def Nmask(tg_ds, s3bkt, indstr, n_workers=2):
    nmc = NmaskClassifier(s3bkt, indstr, n_workers)
    nmask = nmc.compute(tg_ds)
    return nmask  