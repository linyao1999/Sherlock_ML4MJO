import numpy as np
import xarray as xr
from util import rmv_runavg
import multiprocessing

# salloc --nodes 1 --qos interactive --time 03:10:00 --constraint cpu --account=m3312


# salloc --nodes 1 --qos interactive --time 03:10:00 --constraint cpu --account=m3312

# This file filters out the interannual signals of daily anomalies by removing the 120-day running averages. 
# ========= parameters to be set ================
# we will get data anomalies for periods of date_sta to date_end
date_sta = '1979-01-01'
date_end = '2022-12-31'

# this indicates the time range of daily anomaly data file used. 
year_sta = '1979'
year_end = '2022'

# this indicates which years are used to get the annual cycle.
yearclim_sta=1979
yearclim_end=2012

# daily anomaly is stored in this directory.
dirn = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/'

# filtered anomaly is store in this directory.
# NOTE: remember to create the directory. 
data_store = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/'
# ================================================


# =========== do not change codes below ================
# variable used 
varn_list = [
    'olr','u200','u850',
]

data_path_list = [dirn + vari +'.anomaly.' + year_sta + 'to' + year_end + 'based' + str(yearclim_sta) + 'to' + str(yearclim_end) + '.nc' for vari in varn_list]

# define a function to run get_anomalies_1var
def compute_parallel(varn, data_path):
    ds = xr.open_dataset(data_path)
    target_ano = ds[varn]

    print('start compute: ' + varn)
    print('target_ano min: ', target_ano.min().values)
    print('target_ano max: ', target_ano.max().values)

    flt_ano = rmv_runavg(target_ano, window_size=120)  # remove the previous 120-day running averages

    flt_ano_sel = flt_ano.sel(time=slice(date_sta, date_end))

    print('flt_ano_sel min: ', flt_ano_sel.min().values)
    print('flt_ano_sel max: ', flt_ano_sel.max().values)

    # Assign a name to the DataArray
    data_array = flt_ano_sel.rename(varn)

    # Convert the DataArray to a Dataset
    dataset = data_array.to_dataset()

    # Save the Dataset to an NC file
    dataset.to_netcdf(data_store + varn +'.fltano120.' + year_sta + 'to' + year_end + 'based' + str(yearclim_sta) + 'to' + str(yearclim_end) + '.nc', mode='w')

    # Close the Dataset
    dataset.close()
    ds.close()


pool = multiprocessing.Pool(19)

pool.starmap(compute_parallel, zip(varn_list, data_path_list))

# Close the pool of worker processes
pool.close()
pool.join()

exit()