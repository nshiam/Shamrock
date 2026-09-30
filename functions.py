import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"





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
multiprocess.set_start_method("fork", force=True)



def load_snana_directory(path_sim):


    head_files = sorted(glob(os.path.join(path_sim, "*_HEAD.FITS.gz")))
    phot_files = sorted(glob(os.path.join(path_sim, "*_PHOT.FITS.gz")))

    if len(head_files) == 0 or len(phot_files) == 0:
        raise ValueError("No HEAD or PHOT files found.")

    data_h = []
    data_p = []

    # Match HEAD & PHOT files by prefix
    for head_file in head_files:
        prefix = os.path.basename(head_file).replace("_HEAD.FITS.gz", "")

        # # Load HEAD
        # df_head = pd.DataFrame(fits.open(head_file)[1].data)
        # df_head["SNID"] = df_head["SNID"].str.strip()
        # data_h.append(df_head)
    
        t = Table.read(head_file)
        names = [name for name in t.colnames if len(t[name].shape) <= 1]
        df_head = t[names].to_pandas()

        for col in df_head.select_dtypes(include='object'):
            first_valid = df_head[col].dropna().iloc[0] if df_head[col].notna().any() else None
            if isinstance(first_valid, (bytes, bytearray)):
                df_head[col] = df_head[col].str.decode('utf-8').str.strip()
    
        data_h.append(df_head)

        
    for phot_file in phot_files:
        prefix = os.path.basename(phot_file).replace("_PHOT.FITS.gz", "")
       
        # Load PHOT
        data = fits.open(phot_file)[1].data.byteswap().newbyteorder()
        df_phot = pd.DataFrame(data.view(np.recarray))
        if "FLT" in df_phot.columns:
            df_phot.rename(columns={"FLT": "BAND"}, inplace=True)
        df_phot["BAND"] = (
            df_phot["BAND"]
            .str.decode("utf-8")
            .str.strip()
            .str.split("-")
            .str[-1]
        )

        
        data_p.append(table_sncosmo(df_phot))

    return data_h, data_p



def run_id_function(file_id, start, end, data_h=None):

    if file_id == 'NONE':

        if data_h is None or len(data_h) == 0:
            raise ValueError("data_h is empty or not provided.")

        # data_h assumed to be dictionary
        total = pd.concat(data_h, ignore_index=True)
        idd = id_function(total)
        idd = idd[start:end]

    else:

        try:
            ids = pd.read_csv(
                file_id,
                sep=r'\s+',
                compression='gzip',
                comment='#'
            )

        except Exception:
            try:
                ids = pd.read_csv(file_id)

            except Exception:
                try:
                    ids = pd.read_hdf(file_id, key='table')

                except Exception:
                    raise ValueError("Could not read file.")

        idd = id_function(ids)
        idd = idd[start:end]

    return idd


def flagged_photometry(df):
    bad_flags = [8, 16, 32, 64, 128, 256, 512]

    bad_mask = 0
    for flag in bad_flags:
        bad_mask |= flag

    good_rows = (df['PHOTFLAG'] & bad_mask) == 0

    df_clean = df[good_rows]

    # print(f"Removed {len(df) - len(df_clean)} rows with bad flags")
    return df_clean

def estimates_griz_riz(griz, riz):
    griz.rename_column('photoz', 'photoz_griz')
    riz.rename_column('photoz', 'photoz_riz')
    
    # Combine tables column-wise
    t = Table([griz['SNID'], griz['photoz_griz'], riz['photoz_riz']],
              names=('SNID', 'photoz_griz', 'photoz_riz'))
    
    # Create final photoz column
    t['photoz'] = np.where(
        t['photoz_riz'] > 0.9,
        t['photoz_riz'],
        t['photoz_griz']
    )
    
    # Keep only what you need
    final_table = t['SNID', 'photoz']
    return final_table



#function that takes as input the photo data i.e. the data_p1,2,3,4,5 and converts it to the astropy table format. 
#Should run it once for each data_p_ and then only use that sncosmo table.
def table_sncosmo(data_phot):
    # make a astropy table to fit the light curves, we need time, band, flux, flux error, zero point, mag system 
    selected_columns = data_phot[['MJD', 'BAND' , 'FLUXCAL', 'FLUXCALERR', 'PHOTFLAG']]
    # Rename the columns 
    new_column_names = {'MJD': 'mjd', 'BAND': 'band', 'FLUXCAL': 'flux', 'FLUXCALERR': 'fluxerr' } 
    selected_columns.rename(columns=new_column_names, inplace=True)
    # Convert Pandas DataFrame to astropy table
    astropy_table = Table.from_pandas(selected_columns)
    # Replace the bands with the sncosmo band names
    astropy_table['band'] = astropy_table['band'].astype('S4')
    astropy_table['band'][astropy_table['band'] == 'g'] = 'desg'
    astropy_table['band'][astropy_table['band'] == 'r'] = 'desr'
    astropy_table['band'][astropy_table['band'] == 'i'] = 'desi'
    astropy_table['band'][astropy_table['band'] == 'z'] = 'desz'
    astropy_table['band'] = astropy_table['band'].astype(str)
    astropy_table['magsys'] = 'ab'
    astropy_table['zp'] = ZP
    
    return astropy_table


#magnitude function to compute theoretical and observed magnitude for each redshift interval
def mag_comp(z, x0,x1,c, M_trial = -19.35, a=alpha_cosmo, b =beta_cosmo, H_0=70, omega_m=0.3):
    offset = 10.635
    cosmo = FlatLambdaCDM(H0=H_0, Om0= omega_m)
    m_theory = cosmo.distmod(z).value + M_trial
    m_interval = -2.5* np.log10(x0) + a* x1 - b * c + offset
    return  m_theory, m_interval



#quick function that locates the SNIDs of the SN that we are interested in
def id_function(data_p):
    #get the ids of the spectro-z as str
    idd = data_p['SNID'].astype(str)
    idd = idd[ idd != 'nan'] 
    idd = idd.astype(float) ; idd = idd.astype(int) ; idd = idd.astype(str)
    return idd


def truncate_to_sf(x, sig_figs):
    "This is a toy function that I use to check for edge effect systematics. It returns the z of the bin that we need to check"
    if x == 0:
        return 0
    else:
        # Calculate the scaling factor to shift the number to the right
        scale = math.floor(math.log10(abs(x))) + 1
        
        # Truncate by shifting, converting to integer, and then scaling back
        return math.floor(x * 10**(sig_figs - scale)) / 10**(sig_figs - scale)


def check_systematic(z_photo, prec):
    "check for edge degeneracy. Dependin on the precision it changes things. It simply returns a value of 0 or 1 or 2."
    "0 correspnods to ok estimates. 1 corresponds to estimates close to .0000 (lower edge of bin)."
    "2 corresponds to estimates close to .9999 (upper edge of bin)."
    dec_part = np.abs(int(z_photo*10) - z_photo*10)
    dec_part = truncate_to_sf(dec_part, prec)
    limit1 = 10**(-prec)
    check = 0
    if prec ==1 :
        limit2 = 0.9
    elif prec == 2:
        limit2 = 0.99
    elif prec == 3:
        limit2= 0.999
    elif prec == 4:
        limit2= 0.9999    

    if dec_part < 0.5:
        if dec_part < limit1:
            check = 1
    else:
        if dec_part == limit2:
            check=2
    return check
    

def zbins_estimates2(fit_table, SNID, z_min , z_max , dz_width = 0.1, number_of_floating = 2):
    bound = np.round(np.arange(z_min, z_max + dz_width, dz_width), number_of_floating )
    z =[] ; x1=[]; c=[] ; chisqr = []; dof=[] ; x0 = []
    model = sncosmo.Model(source=salt_model)
    if not np.any(fit_table['flux'] / fit_table['fluxerr'] > SNR):
        if not np.any(fit_table['flux'] / fit_table['fluxerr'] > 3):  
            fail = (SNID,
                    np.array([1e8] * (len(bound)-1)),  # chisqr
                    np.array([-999] * (len(bound)-1)), # z
                    np.array([1e8]  * (len(bound)-1)), # x0
                    np.array([-999] * (len(bound)-1)), # x1
                    np.array([-999] * (len(bound)-1)), # c
                    np.array([1e-8] * (len(bound)-1))  # dof
                   )
            return fail
    for i in range(len(bound)-1):   
        try:
            # Use a context manager to catch specific warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Catch all warnings
                
                # Attempt to fit the light curve
                r, fm = sncosmo.fit_lc(fit_table, model, ['z', 't0', 'x0', 'x1', 'c'] ,
                                       bounds={'z': (bound[i], bound[i + 1]), 'x1':(-5,5), 'c': (-0.5,0.5),'x0': (0.0,0.1) }, minsnr = SNR)
                # No warnings, so append results
                z.append(r.parameters[0]) ; x0.append(r.parameters[2])
                x1.append(r.parameters[3]) ; c.append(r.parameters[4]) 
                chisqr.append(r.chisq)     ; dof.append(r.ndof)
        except (RuntimeError, DataQualityError): 
            z.append(-999) ; x0.append(1e8)
            x1.append(-999) ; c.append(-999)  
            chisqr.append(1e8) ; dof.append(1e-8)

    chisqr = np.array(chisqr) ; dof = np.array(dof)
    z = np.array(z)   
    x1 = np.array(x1) ; c = np.array(c) 

    return SNID, chisqr, z, x0, x1, c, dof

# function to identify the ids to use 
def lc_fitting_step1_2(data_head, table_for_sncosmo, SNID, bands, z_min , z_max, dz_width):
    id = np.where(data_head['SNID'].str.strip()== SNID)[0]  #SNID must be a string
    if len(id) > 0:
        id=id[0]
        min_phot , max_phot = data_head['PTROBS_MIN'][id], data_head['PTROBS_MAX'][id]
        fit_table = table_for_sncosmo[min_phot-1:max_phot] #maybe it should be min_phot -1
        if PHOTFLAG == 'TRUE':
            fit_table = flagged_photometry(fit_table) #!!!
        fit_table.remove_column('PHOTFLAG') #!!!
        if len(fit_table) > 10:
            ref_mjd = fit_table['mjd'][10]
            fit_table = fit_table[fit_table['mjd'] < (ref_mjd + 200)]
        else:
            # not enough data points, so skip
            return None
        if bands == 'riz':
            fit_table  = fit_table[fit_table['band'] != 'desg']
        
        zbins_es = zbins_estimates2(fit_table, SNID, z_min, z_max, dz_width)
        return  zbins_es
    return  None

#simple loop to run using the lc_fitting_step1
def run_es2(idd, data_h, table_p, bands, z_min , z_max, dz_width):   
    id_yr1 = idd
    zbins_es =[]
    for i in id_yr1:
        zbins_ess = lc_fitting_step1_2(data_h, table_p, i, bands, z_min , z_max, dz_width)
        if zbins_ess is not None:  # Check if the result is valid
            zbins_es.append(zbins_ess)
    return zbins_es




# CHANGE VALUES TO CHANGE PRIORS: x1, c, and Delta_m. 
def loop_for_fitting(zbins_es, x1_l, x1_u, c_l, c_u, m_d, z_max, z_min):
    SNID = zbins_es[0] ; chisqr = zbins_es[1] ; chisqr = np.array(chisqr) 
    z = zbins_es[2]   ; z = np.array(z)  
    x0 = zbins_es[3] ; x0 = np.array(x0)
    x1 =  zbins_es[4] ; x1 = np.array(x1) 
    c =  zbins_es[5] ; c = np.array(c)  
    dof = zbins_es[6] ; dof = np.array(dof)
    r_chisqr = (chisqr/dof)
    min_cond = r_chisqr
    z_initial = z
    while True:
        index = np.argmin(min_cond)
        m_th , m_int = mag_comp(z[index], x0[index],x1[index],c[index], M_trial = -19.35, a=alpha_cosmo, b = beta_cosmo, H_0=70, omega_m=0.3)
        m_th = np.array(m_th) ; m_int = np.array(m_int)
        check = check_systematic(z[index], 3)
        if ((z[index] > z_min + 0.05) and (z[index] < z_max - 0.0001 ) and (x1[index] > x1_l) and (x1[index] < x1_u) and (c[index] > c_l) and (c[index] < c_u)  
            and (np.abs(m_th - m_int) < m_d)):
            if check == 0:
                z_b = z[index] 
                break
            elif check == 1: 
                interval_index = int(z[index] *10) 
                if np.abs(z_initial[interval_index] - z_initial[interval_index-1]) < 0.04:
                    z_b = z[index] 
                    break 
                
            elif check ==2:
                interval_index = int(z[index] *10) 
                if (interval_index+1) < len(z_initial):
                    if np.abs(z_initial[interval_index] - z_initial[interval_index+1]) <0.04:
                        z_b = z[index] 
                        break 
            
        rest_ind = np.where(min_cond > min_cond[index])[0]
        min_cond = min_cond[rest_ind]
        z = z[rest_ind] 
        x0 = x0[rest_ind]
        x1 = x1[rest_ind] 
        c = c[rest_ind] 
        if len(min_cond) < 1:
            z_b = -999 
            break
        
    return SNID, z_b
    

#step 2 simply run the fitting for a single value!
def lc_fitting_step2(zbins_es, x1_l, x1_u, c_l, c_u, m_d, z_max, z_min):
    if zbins_es is not None:
        SNID, z_b, = loop_for_fitting(zbins_es,x1_l, x1_u, c_l, c_u, m_d, z_max, z_min)
        return SNID, z_b

# run loop for all ids that you want
def run_fit(zbins_es, x1_l, x1_u, c_l, c_u, m_d, z_max, z_min): 
    z_t = []  
    id = []
    for i in range(len(zbins_es)):
        ssnid, z_b  = lc_fitting_step2(zbins_es[i],x1_l, x1_u, c_l, c_u, m_d, z_max, z_min)
        if z_b is not None:  # Check if the result is valid
            z_t.append(z_b)
            id.append(ssnid)
    photoz_table = {'SNID': id , 'photoz': z_t}
    table = Table(photoz_table)
    
    return table



def chi2_estimate(fit_table, SNID, photo_z, type_z):
    z =[] ; z_err=[]; t0 =[] ; t0_err =[] ; x0=[] ; x0_err=[]; x1=[]; x1_err=[]; c=[] ; c_err=[] ;  chisqr = []; dof=[]
    model = sncosmo.Model(source=salt_model)
    fit_success = False  
    if type_z == 'PHOTOZ':
        z_ranges = [(0.05, 0.05), (0.03, 0.03), (0.03, 0.02), (0.02, 0.01),(0.01,0.02), (0.01,0.01), (0.002, 0.001), (0.001,0.002), (0.003, 0.001), (0.0, 0.0)]

    for z_add in z_ranges:
        z_lower, z_upper = photo_z - z_add[0], photo_z + z_add[1]
        try:
            r, f_m = sncosmo.fit_lc(fit_table, model,['z', 't0', 'x0', 'x1', 'c'], 
                                    bounds={'z': (z_lower, z_upper), 'x1':(-5,5), 'c': (-0.5,0.5), 'x0': (0.0,0.1)}, minsnr=SNR)
            fit_success = True
            break
        except Exception:
            continue
    if fit_success:
        z.append(r.parameters[0]);   z_err.append(r.errors['z'])
        t0.append(r.parameters[1]); t0_err.append(r.errors['t0'])
        x0.append(r.parameters[2]); x0_err.append(r.errors['x0'])
        x1.append(r.parameters[3]); x1_err.append(r.errors['x1'])
        c.append(r.parameters[4]);  c_err.append(r.errors['c'])
        chisqr.append(r.chisq);     dof.append(r.ndof)
    else:
        # append failure values
        z.append(-999); z_err.append(-999)
        t0.append(-999); t0_err.append(-999)
        x0.append(-999); x0_err.append(-999)
        x1.append(-999); x1_err.append(-999)
        c.append(-999); c_err.append(-999)
        chisqr.append(-999); dof.append(-999)

    return SNID, z, z_err, t0, t0_err, x0, x0_err, x1, x1_err, c, c_err, chisqr, dof


# function to identify the ids to use 
def lc_chi2_step1_2(data_head, table_for_sncosmo, SNID, photo_z, type_z):
    id = np.where(data_head['SNID'].str.strip()== SNID)[0]  #SNID must be a string
    if len(id) > 0:
        id=id[0]
        min_phot , max_phot = data_head['PTROBS_MIN'][id], data_head['PTROBS_MAX'][id]
        fit_table = table_for_sncosmo[min_phot-1:max_phot]
        if PHOTFLAG == 'TRUE':
            fit_table = flagged_photometry(fit_table) #!!!
        fit_table.remove_column('PHOTFLAG') #!!!
        if len(fit_table['mjd']) > 10 :
            fit_table  = fit_table[fit_table['mjd']  < (fit_table['mjd'][10] +200) ]
            zbins_es = chi2_estimate(fit_table, SNID, photo_z, type_z)
        else: 
            zbins_es= None
        return  zbins_es
    return  None

#simple loop to run using the lc_fitting_step1
def run_chi2_es(data, data_h, table_p, type_z = 'PHOTOZ'):   
    id = np.array(id_function(data))
    if type_z == 'PHOTOZ':
        photo_z = np.array(data['photoz'])
    elif type_z == 'SPECZ':
        photo_z = np.array(data['zHEL'])
    zbins_es =[]
    for i in range(len(id)):
        zbins_ess = lc_chi2_step1_2(data_h, table_p, id[i] , photo_z[i], type_z)
        if zbins_ess is not None:  # Check if the result is valid
            zbins_es.append(zbins_ess)
    return zbins_es


def make_table_photoz_pickle(pickle):
    id=[] ; z=[] ; z_err =[] ; t0 =[] ; t0_err =[]
    x0 =[] ; x0_err =[] ; x1 =[] ; x1_err =[]
    c=[] ; c_err =[] ; chisqr = [] ; dof = []
    for i in range(len(pickle)):
        id.append(pickle[i][0])
        z.append(pickle[i][1][0]) ; z_err.append(pickle[i][2][0])
        t0.append(pickle[i][3][0]) ; t0_err.append(pickle[i][4][0])
        x0.append(pickle[i][5][0]) ; x0_err.append(pickle[i][6][0])
        x1.append(pickle[i][7][0]) ; x1_err.append(pickle[i][8][0])
        c.append(pickle[i][9][0]) ; c_err.append(pickle[i][10][0])
        chisqr.append(pickle[i][11][0]) ; dof.append(pickle[i][12][0])
        
    tb = {'SNID': id , 'photoz_sncosmo': z, 'photoz_err_sncosmo': z_err,     
          't0_photoz_sncosmo': t0, 't0_err_photoz_sncosmo': t0_err, 'x0_photoz_sncosmo': x0, 'x0_err_photoz_sncosmo': x0_err, 
          'x1_photoz_sncosmo': x1, 'x1_err_photoz_sncosmo': x1_err, 'c_photoz_sncosmo': c, 'c_err_photoz_sncosmo': c_err, 
         'chisqr_sncosmo' : chisqr  , 'dof_sncosmo': dof }
    table = Table(tb)
    return table



def run_shiamrock(proc_id, data_h, table_p, idd, z_griz_min, z_griz_max, z_riz_min, z_riz_max, dz_width):
    print(f'Process {proc_id} running with IDs: {idd}...')
    
    griz = []
    for dh, tp in zip(data_h, table_p):
        griz += run_es2(idd, dh, tp,
                        bands='griz',
                        z_min= z_griz_min,
                        z_max= z_griz_max, dz_width = dz_width)
    
    #pickle dump here
    filename_griz = os.path.join(run_dir, f"photoz_estimates_snr_{SNR}_griz_{start}_{end}_{proc_id}.pkl")
    
    with open(filename_griz, 'wb') as f:
        pickle.dump(griz, f)
    
    
    riz = []
    for dh, tp in zip(data_h, table_p):
        riz += run_es2(idd, dh, tp,
                        bands='riz',
                        z_min= z_riz_min,
                        z_max= z_riz_max  , dz_width = dz_width)  
    
    
    
    #pickle dump here
    filename_riz = os.path.join(run_dir, f"photoz_estimates_snr_{SNR}_riz_{start}_{end}_{proc_id}.pkl")
    
    with open(filename_riz, 'wb') as f:
        pickle.dump(riz, f)
    
    
    z_griz = run_fit(griz, x1_l, x1_u, c_l, c_u, m_d, z_griz_max, z_griz_min)
    
    z_riz = run_fit(riz, x1_l, x1_u, c_l, c_u, m_d, z_riz_max, z_riz_min)
    
    dd = estimates_griz_riz(z_griz, z_riz)
    
    
    
    dd = dd[dd['photoz'] >0 ]
    est = []
    for dh, tp in zip(data_h, table_p):
        est += run_chi2_es(dd, dh, tp, type_z)

    df = make_table_photoz_pickle(est).to_pandas()
    #save it here too.
    filename_df = os.path.join(run_dir, f"shiamrock_trial_{proc_id}.csv")
    df.to_csv(filename_df, sep = ' ', index = False )

    return 




def split_paths(a, n):
    """
    Splits images evenly among requested processes
    """

    k, m = divmod(len(a), n)

    return (a[i * k +  min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

