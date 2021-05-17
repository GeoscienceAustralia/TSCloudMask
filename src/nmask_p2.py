import tensorflow as tf
import tensorflow.keras.models as models
import numpy as np
import sys
import os






def std_by_paramters(data, rs, msarr):
    ntr=data.shape[1]
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=msarr[i]
        std=msarr[i+ntr]
        clm=(clm-mu)/(rs*std)
        data[:,i]=clm
        
    return data



# 

def main():
        
    param=sys.argv
    argc = len(param)

    if ( argc != 5 ):

        print("Usage: python3 nmask_p2.py dirc loc_str indpred")  
        print("dirc: Input/Output data directory")
        print("loc_str: location string for the output file name")
        print("modeldirc: directory where the NN model and standalised parameter files located ") 
        print("modelname: file name of the NN model ") 

    # Input/Output data directory
    datadirc = param[1]

    # location string in the output filename
    loc_str = param[2]
    
    # suffix of long term mean of indices data files  
    modeldirc = param[3]
    
    # suffix of long term mean of indices data files  
    modelname = param[4]
  
    outdirc = datadirc
    
    
    parafilename=modeldirc+'/'+modelname+'_standarise_parameters.npy'
    norm_paras = np.load(parafilename)
    modelfilename=modeldirc+'/'+modelname
    model = models.load_model(modelfilename)



    timebandsfname = datadirc + '/' + loc_str + '_timebandnames.npy'  
    tsbandnames = np.load(timebandsfname)



    for tbname in tsbandnames:

        datafname = datadirc + '/' + loc_str + '_'+tbname+'_ipdata.npy'
        ipdata = np.load(datafname)

        ss = ipdata.size
        pnum = int(ss/12)

        ipdata = ipdata.reshape(pnum, 12)
        ipdata = std_by_paramters(ipdata, 2, norm_paras)

        print(tbname, ipdata.shape)

        mixtures=model.predict(ipdata)
        print(mixtures.shape)

        print(mixtures)
        mixtures = mixtures.flatten()
        mixfname = outdirc + '/' + loc_str + '_'+tbname+'_predict'
        np.save(mixfname, mixtures)


if __name__ == '__main__':
    main()



