import argparse
from config import ShamrockConfig


def run(config):
    start_clock = time.perf_counter()
    # salt_model = 'salt3'
    salt_model = sncosmo.SALT3Source(modeldir='SALT3.DOVEKIE/')
    data_h, table_p = load_snana_directory(file_path)
    total = pd.concat(data_h, ignore_index=True)

    idd = run_id_function(prob, config.START, end, data_h=data_h)

    
    if pickle_files == 'FALSE': 
        split_ids = list(split_paths(list(idd), ncores))
        jobs, proc_ids = [], []
        for proc_id in range(ncores):
            if split_ids[proc_id]:
                proc = multiprocess.Process(target=run_shiamrock, 
                                            args=(proc_id, data_h, table_p, split_ids[proc_id], z_griz_min, z_griz_max, z_riz_min, z_riz_max, dz_width))
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
    
            
            filename_griz = os.path.join(run_dir,f"photoz_estimates_snr_{SNR}_griz_{start}_{end}_{proc_id}.pkl")
            with open(filename_griz, "rb") as f:
                all_griz += pickle.load(f)
        
            filename_riz = os.path.join(run_dir,f"photoz_estimates_snr_{SNR}_riz_{start}_{end}_{proc_id}.pkl")
            with open(filename_riz, "rb") as f:
                all_riz += pickle.load(f)
        
            filename_df = os.path.join(run_dir, f"shiamrock_trial_{proc_id}.csv")
            df_part = pd.read_csv(filename_df,
                                  sep=" ")
            all_dfs.append(df_part)
        
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all = df_all[df_all['photoz_sncosmo'] > 0]
        
        filename_griz_all = os.path.join(run_dir,f"{save_file}_photoz_estimates_snr_{SNR}_griz_{start}_{end}_ALL.pkl")
        with open(filename_griz_all, "wb") as f:
            pickle.dump(all_griz, f)
    
        filename_riz_all = os.path.join(run_dir,f"{save_file}_photoz_estimates_snr_{SNR}_riz_{start}_{end}_ALL.pkl")
        with open(filename_riz_all, "wb") as f:
            pickle.dump(all_riz, f)
    
        filename_df_all = os.path.join(run_dir,f"{save_file}_shiamrock_{start}_{end}_ALL.csv")
        df_all.to_csv(filename_df_all, sep=" ", index=False)
        
        
        for proc_id in proc_ids:
        
            os.remove(os.path.join(run_dir,f"photoz_estimates_snr_{SNR}_griz_{start}_{end}_{proc_id}.pkl"))
            os.remove(os.path.join(run_dir,f"photoz_estimates_snr_{SNR}_riz_{start}_{end}_{proc_id}.pkl"))
            os.remove(os.path.join(run_dir,f"shiamrock_trial_{proc_id}.csv"))

    elif pickle_files == "TRUE":
        
        
        filename_griz = os.path.join(run_dir,f"{save_file}_photoz_estimates_snr_{SNR}_griz_{start}_{end}_ALL.pkl")
        filename_riz = os.path.join(run_dir,f"{save_file}_photoz_estimates_snr_{SNR}_riz_{start}_{end}_ALL.pkl")


        with open(filename_griz, "rb") as f:
            griz = pickle.load(f)
            
        with open(filename_riz, "rb") as f:
            riz = pickle.load(f)

        z_griz = run_fit(griz, x1_l, x1_u, c_l, c_u, m_d, z_griz_max, z_griz_min)
        z_riz = run_fit(riz, x1_l, x1_u, c_l, c_u, m_d, z_riz_max, z_riz_min)
    
        dd = estimates_griz_riz(z_griz, z_riz)
    
    
    
        dd = dd[dd['photoz'] >0 ]
        est = []
        
        for dh, tp in zip(data_h, table_p):
            est += run_chi2_es(dd, dh, tp, type_z)

        df = make_table_photoz_pickle(est).to_pandas()
        #save it here too.
        filename_df = os.path.join(run_dir, f"shiamrock_a_{alpha_cosmo}_b_{beta_cosmo}_Dm_{m_d}_{x1_l}_x1_{x1_u}_{c_l}_c_{c_u}_trial.csv")
        df.to_csv(filename_df, sep = ' ', index = False )


    
    end_clock = time.perf_counter()
    print('TOTAL TIME THAT WE CARE: ', end_clock - start_clock, 'seconds OR', (end_clock - start_clock)/60, 'min' )


    df = pd.read_csv(os.path.join(run_dir,f"{save_file}_shiamrock_{start}_{end}_ALL.csv"), sep = ' ')
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
    		    
    
    if HEADER_OVERRIDE == 'TRUE':
        new.to_csv(os.path.join(run_dir,f"{save_file}_shiamrock_HEADER_OVERRIDE.dat"), sep = ' ' , index = False )
        


  
    return





if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-FILE_PATH")
    parser.add_argument("--NCORES", help="Set number of cores to use")
    parser.add_argument("--START", help="Set number of cores to use")
    
    args = parser.parse_args() 

    config = ShamrockConfig()
  
    for arg_name, arg_value in vars(args).items():
        print(arg_name, arg_value)

        if not arg_value:
          continue
          
        setattr(config, arg_name, arg_value)

    run(config)




