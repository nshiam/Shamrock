import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"




import argparse
from config import ShamrockConfig
import functions
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats
import sncosmo
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.io import fits
import time
from astropy.table import Table
import warnings
import pickle
import math
from sncosmo.fitting import DataQualityError
from glob import glob
import sys
import multiprocess

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", message="Dropping following bands from data:.*", category=RuntimeWarning)
warnings.filterwarnings("ignore")


def str_to_bool(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError(
            "Expected True or False")

    return value.lower() == "true"

def run(config):
    multiprocess.set_start_method("fork", force=True)
  
    start_clock = time.perf_counter()
    print()
    print('--- Reading the Light Curve files ---')
    data_h, table_p = functions.load_snana_directory(config)
    total = pd.concat(data_h, ignore_index=True)
  
    print()
    print('--- Light Cruve files are fine ---')

    if config.END == None:
        config.END = len(total)
    idd = functions.run_id_function(config, data_h=data_h)

    print()
    print('--- IDs are selected ---')

    
    if config.PICKLE_FILES == False:
        print()
        print('--- Shamrock implemented - Approx. Time:', 
              11 * config.END / config.NCORES ,' seconds or ', f"{12 * config.END / (config.NCORES * 60):.2f}"  ,' minutes ---')
        print()
        split_ids = list(functions.split_paths(list(idd), config.NCORES))
        jobs, proc_ids = [], []
        for proc_id in range(config.NCORES):
            if split_ids[proc_id]:
                proc = multiprocess.Process(target=functions.run_shiamrock, 
                                            args=(config, proc_id, data_h, table_p, split_ids[proc_id]))
                jobs.append(proc)
                proc_ids.append(proc_id)
    
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
    
        all_griz = []
        all_riz = []
        all_dfs = []
        
        for proc_id in proc_ids:
    
            
            filename_griz = os.path.join(config.RUN_DIR,f"photoz_estimates_snr_{config.SNR}_griz_{config.START}_{config.END}_{proc_id}.pkl")
            with open(filename_griz, "rb") as f:
                all_griz += pickle.load(f)
        
            filename_riz = os.path.join(config.RUN_DIR,f"photoz_estimates_snr_{config.SNR}_riz_{config.START}_{config.END}_{proc_id}.pkl")
            with open(filename_riz, "rb") as f:
                all_riz += pickle.load(f)
        
            filename_df = os.path.join(config.RUN_DIR, f"shiamrock_trial_{proc_id}.csv")
            df_part = pd.read_csv(filename_df,
                                  sep=" ")
            all_dfs.append(df_part)
        
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all = df_all[df_all['photoz_sncosmo'] > 0]
        
        filename_griz_all = os.path.join(config.RUN_DIR, 
                                         f"{config.SAVE_FILE}_photoz_estimates_snr_{config.SNR}_griz_{config.START}_{config.END}_ALL.pkl")
        with open(filename_griz_all, "wb") as f:
            pickle.dump(all_griz, f)
    
        filename_riz_all = os.path.join(config.RUN_DIR,
                                        f"{config.SAVE_FILE}_photoz_estimates_snr_{config.SNR}_riz_{config.START}_{config.END}_ALL.pkl")
        with open(filename_riz_all, "wb") as f:
            pickle.dump(all_riz, f)
    
        filename_df_all = os.path.join(config.RUN_DIR,f"{config.SAVE_FILE}_shiamrock_{config.START}_{config.END}_ALL.csv")
        df_all.to_csv(filename_df_all, sep=" ", index=False)
        
        
        for proc_id in proc_ids:
        
            os.remove(os.path.join(config.RUN_DIR,f"photoz_estimates_snr_{config.SNR}_griz_{config.START}_{config.END}_{proc_id}.pkl"))
            os.remove(os.path.join(config.RUN_DIR,f"photoz_estimates_snr_{config.SNR}_riz_{config.START}_{config.END}_{proc_id}.pkl"))
            os.remove(os.path.join(config.RUN_DIR,f"shiamrock_trial_{proc_id}.csv"))

    elif config.PICKLE_FILES == True:
        
        
        filename_griz = os.path.join(config.RUN_DIR,
                                     f"{config.SAVE_FILE}_photoz_estimates_snr_{config.SNR}_griz_{config.START}_{config.END}_ALL.pkl")
        filename_riz = os.path.join(config.RUN_DIR,f"{config.SAVE_FILE}_photoz_estimates_snr_{config.SNR}_riz_{config.START}_{config.END}_ALL.pkl")


        with open(filename_griz, "rb") as f:
            griz = pickle.load(f)
            
        with open(filename_riz, "rb") as f:
            riz = pickle.load(f)

        z_griz = functions.run_fit(griz, config.X1_LOWER, config.X1_UPPER,
                                   config.C_LOWER, config.C_UPPER, config.DELTA_MAG, config.GRIZ_MAX, config.GRIZ_MIN)
        z_riz = functions.run_fit(riz, config.X1_LOWER, config.X1_UPPER, 
                                  config.C_LOWER, config.C_UPPER, config.DELTA_MAG, config.RIZ_MAX, config.RIZ_MIN)
    
        temp = functions.estimates_griz_riz(z_griz, z_riz)
    
    
    
        temp = temp[temp['photoz'] >0 ]
        est = []
        
        for dh, tp in zip(data_h, table_p):
            est += functions.run_chi2_es(temp, dh, tp, config.TYPE_Z)

        df = functions.make_table_photoz_pickle(est).to_pandas()
        #save it here too.
        filename_df = os.path.join(config.RUN_DIR,
                                f"shiamrock_a_{config.ALPHA_COSMO}_b_{config.BETA_COSMO}_Dm_{config.DELTA_MAG}_{config.X1_LOWER}_x1_{config.X1_UPPER}_{config.C_LOWER}_c_{config.C_UPPER}_trial.csv")
        df.to_csv(filename_df, sep = ' ', index = False )


    
    end_clock = time.perf_counter()
    print('TOTAL TIME THAT WE CARE: ', end_clock - start_clock, 'seconds OR', (end_clock - start_clock)/60, 'min' )


    df = pd.read_csv(os.path.join(config.RUN_DIR,f"{config.SAVE_FILE}_shiamrock_{config.START}_{config.END}_ALL.csv"), sep = ' ')
    df = df[df['photoz_sncosmo'] > 0.05]
    
    total = pd.concat(data_h, ignore_index=True)
    total['SNID'] = total['SNID'].astype(int)
    n = pd.merge(total, df, on = 'SNID', how = 'left')
    
    n['SNID'] = n['SNID'].astype(int)
    n.rename(columns = {'SNID' : 'CID'}, inplace = True)
    n = n[['CID', 'photoz_sncosmo', 'photoz_err_sncosmo']]
    n['photoz_sncosmo'] = n['photoz_sncosmo'].fillna(-999.00)
    n['photoz_err_sncosmo'] = n['photoz_err_sncosmo'].fillna(-999.00)
    
    
    
    new = total[['SNID', 'REDSHIFT_HELIO', 'HOSTGAL_SPECZ', 'REDSHIFT_FINAL', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_SPECZ_ERR','REDSHIFT_FINAL_ERR' ]]
    new['VARNAMES:'] = 'SN:'
    new = new[['VARNAMES:'] + list(new.columns.drop('VARNAMES:'))]
    
    
    new['REDSHIFT_HELIO'] = n['photoz_sncosmo']
    new['REDSHIFT_HELIO_ERR'] = n['photoz_err_sncosmo']
    new['REDSHIFT_FINAL'] = n['photoz_sncosmo']
    new['REDSHIFT_FINAL_ERR'] =n['photoz_err_sncosmo']
    new['HOSTGAL_SPECZ'] = n['photoz_sncosmo']
    new['HOSTGAL_SPECZ_ERR'] = n['photoz_err_sncosmo']
    		    
    
    if config.HEADER_OVERRIDE == True:
        new.to_csv(os.path.join(config.RUN_DIR,f"{config.SAVE_FILE}_shiamrock_HEADER_OVERRIDE.dat"), sep = ' ' , index = False )
        


  
    return



parser = argparse.ArgumentParser()

parser.add_argument("--FILE_PATH",type=str,default=ShamrockConfig.FILE_PATH, help="Directory where HEAD and PHOT files exist")
parser.add_argument("--NCORES", type=int, default=ShamrockConfig.NCORES, help="Number of cores to use")
parser.add_argument("--HEADER_OVERRIDE", type=str_to_bool, 
                    default=ShamrockConfig.HEADER_OVERRIDE, help="Override header information (True or False)")
parser.add_argument("--START", type=int, default=ShamrockConfig.START, help="Starting index")
parser.add_argument("--END", type=int, default=ShamrockConfig.END, help="Ending index")
parser.add_argument("--ZERO_POINT", type=float, default=ShamrockConfig.ZERO_POINT, help="Zero point to use")
parser.add_argument("--SNR", type=float, default=ShamrockConfig.SNR, help="SNR for sncosmo",)
parser.add_argument("--ID_FILE", type=str, default=ShamrockConfig.ID_FILE, help="File containing IDs" )
parser.add_argument("--PHOTFLAG", type=str_to_bool, default=ShamrockConfig.PHOTFLAG, help="Use bitmap flags for photometry (True or False)")
parser.add_argument("--GRIZ_MIN", type=float, default=ShamrockConfig.GRIZ_MIN, help="Minimum GRIZ value")
parser.add_argument("--GRIZ_MAX", type=float, default=ShamrockConfig.GRIZ_MAX, help="Maximum GRIZ value")
parser.add_argument("--RIZ_MIN", type=float, default=ShamrockConfig.RIZ_MIN, help="Minimum RIZ value")
parser.add_argument("--RIZ_MAX", type=float, default=ShamrockConfig.RIZ_MAX, help="Maximum RIZ value")
parser.add_argument("--ZBIN_WIDTH", type=float, default=ShamrockConfig.ZBIN_WIDTH, help="Width of redshift bins")
parser.add_argument("--X1_LOWER", type=float, default=ShamrockConfig.X1_LOWER, help="Lower x1 limit")
parser.add_argument("--X1_UPPER", type=float, default=ShamrockConfig.X1_UPPER, help="Upper x1 limit")
parser.add_argument("--C_LOWER", type=float, default=ShamrockConfig.C_LOWER, help="Lower colour limit")
parser.add_argument("--C_UPPER", type=float, default=ShamrockConfig.C_UPPER, help="Upper colour limit")
parser.add_argument("--DELTA_MAG", type=float, default=ShamrockConfig.DELTA_MAG, help="Magnitude difference threshold")
parser.add_argument("--ALPHA_COSMO", type=float, default=ShamrockConfig.ALPHA_COSMO, help="Cosmological alpha parameter")
parser.add_argument("--BETA_COSMO", type=float, default=ShamrockConfig.BETA_COSMO, help="Cosmological beta parameter")
parser.add_argument("--PICKLE_FILES", type=str_to_bool, 
                    default=ShamrockConfig.PICKLE_FILES, help="Whether to use pre-computed pickle files (True or False)")

# WILL DELETE THESE
parser.add_argument("--SAVE_FILE", type=str, default=None, help="Starting name")
parser.add_argument("--RUN_DIR", type=str, default=ShamrockConfig.FILE_PATH, help="Current running directory")
parser.add_argument("--TYPE_Z", type=str, default=ShamrockConfig.TYPE_Z, help="Dummy" )




if __name__=="__main__":

    
    args = parser.parse_args() 
    if args.RUN_DIR is None:
        args.RUN_DIR = args.FILE_PATH
    if args.SAVE_FILE is None:
        args.SAVE_FILE = os.path.basename(os.path.normpath(args.FILE_PATH))  

    config = ShamrockConfig()
    print()
    print('--- The Configuration that is being used is: ---')
    print()
    for arg_name, arg_value in vars(args).items():
        print(arg_name, arg_value)

        if not arg_value:
          continue
          
        setattr(config, arg_name, arg_value)

    run(config)





