
import numpy as np
import sys
import os



import datacube
import rasterio
from multiprocessing import Pool
import xarray as xr
from datacube.utils.cog import write_cog

sys.path.append('../lib')
import tsmask_func as tsf
import testpair as cym
import dea_datahandling as ddh


def std_by_paramters(data, rs, msarr):
    ntr=data.shape[1]
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=msarr[i]
        std=msarr[i+ntr]
        clm=(clm-mu)/(rs*std)
        data[:,i]=clm
        
    return data


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


 
    
    return s2_ds



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


def cal_indices(blue, green, red, nir, swir1, swir2):
    
    scale = 10000.0
    
    blue = blue.flatten()
    green = green.flatten()
    red = red.flatten()
    nir = nir.flatten()
    swir1 = swir1.flatten()
    swir2 = swir2.flatten()
    
    ivd = -0.0999
    
    ss = blue.size
    
    mask = np.ones(ss, np.uint8)
    
    blue_t = blue/scale
    green_t = green/scale
    red_t = red/scale
    nir_t = nir/scale
    swir1_t = swir1/scale
    swir2_t = swir2/scale
    
    mask[blue<=ivd] = 0
    mask[green<=ivd] = 0
    mask[red<=ivd] = 0
    mask[nir<=ivd] = 0
    mask[swir1<=ivd] = 0
    mask[swir2<=ivd] = 0
    
    scom = (2*nir_t+1)*(2*nir_t+1) - 8*(nir_t - red_t)
    mv = (green_t + red_t + blue_t) / 3
    
    mask[scom<0] = 0
    mask[mv == 0] = 0
    mask[green_t+swir1_t == 0] = 0
    
    
    

    sa = (blue_t+green_t+red_t+nir_t+swir1_t+swir2_t)/6
    mndwi = ((green_t - swir1_t) / (green_t + swir1_t))
    msavi = (2 * nir_t + 1 -np.sqrt(scom))/2
    whi = np.fabs((blue_t - mv)/mv) + np.fabs((green_t - mv)/mv) + np.fabs((red_t - mv)/mv)

  
    return sa, mndwi, msavi, whi, mask


    

def main():
        
    param=sys.argv
    argc = len(param)

    if ( argc != 5 ):

        print("Usage: python3 nmask_p3.py datadirc outdirc loc_str indfile")  

        print("datadirc: Input data directory")
        print("outdirc: output directory")

        print("loc_str: location string for the output file name")
        print("indfile: filename of long term mean of one of the 4 indice")
        
        exit()
        
        
   

    # Input data directory
    datadirc = param[1]
    
    # Output directory
    outdirc = param[2]
    
    # location string in the output filename
    loc_str = param[3]
    
    # filename of long term mean of one of the 4 indice
    indfile = param[4]
    
    
 
      

    comm ='mkdir -p '+ outdirc 
    os.system(comm)


    dbs=xr.open_rasterio(indfile)
    
    timebandsfname = datadirc + '/' + loc_str + '_timebandnames.npy'  
    tsbandnames = np.load(timebandsfname)

    

    irow = dbs['y'].size
    icol = dbs['x'].size
   

    

  
    
    for tbname in tsbandnames:

        onetsmask = np.zeros(irow*icol, dtype = np.uint8)

        mixfname = datadirc + '/' + loc_str + '_'+tbname+'_predict.npy'
        if (os.path.isfile(mixfname)):
                
            maskfname = datadirc + '/' + loc_str + '_'+tbname+'_ipmask.npy'
            mask = np.load(maskfname)

            mixtures = np.load(mixfname)
            ss = mixtures.size
            vpnum = int(ss/3)
            mixtures = mixtures.reshape(vpnum, 3)

            vdmask = np. argmax(mixtures, axis = 1) + 1

            #print(vdmask.shape)
            print(tbname)


            onetsmask[mask==1] = vdmask  
            onetsmask = onetsmask.reshape(irow,icol)
            onetsmask = cym.spatial_filter_v2(onetsmask)
            dbs.data[0] = onetsmask
            outfname = outdirc+'/'+loc_str+'_'+tbname+'_nmask-cog.tif'
            write_cog(geo_im = dbs, fname = outfname, overwrite = True)

        
        
    
        
        
if __name__ == '__main__':
    main()