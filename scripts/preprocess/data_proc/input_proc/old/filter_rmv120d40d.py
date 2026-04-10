import numpy as np
import xarray as xr
from util.MJO_indices_util import rmv_runavg
import multiprocessing

# This file filters out the interannual signals of daily anomalies by removing the 120-day running averages. 
# ========= parameters to be set ================
# we will get data anomalies for periods of date_sta to date_end
date_sta = '1979-01-01'
date_end = '2023-12-31'

# this indicates the time range of daily anomaly data file used. 
year_sta = '1978'
year_end = '2023'

# this indicates which years are used to get the annual cycle.
yearclim_sta=1979
yearclim_end=2012

# daily anomaly is stored in this directory.
dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/anomaly/'

# filtered anomaly is store in this directory.
# NOTE: remember to create the directory. 
data_store = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120_40/'
# ================================================


# =========== do not change codes below ================
# variable used 
varn_list = [
    # 'olr'
    'olr','prep','tcwv',
    'u200', 'u500','u850',
    'v200','v500','v850',
    'q200','q500','q850',
    'T200','T500','T850',
    'Z200','Z500','Z850',
    'sst',
]

data_path_list = [dirn + vari +'.anomaly.' + year_sta + 'to' + year_end + 'based' + str(yearclim_sta) + 'to' + str(yearclim_end) + '.nc' for vari in varn_list]

# define a function to run get_anomalies_1var
def compute_parallel(varn, data_path):
    ds = xr.open_dataset(data_path)
    target_ano = ds[varn]

    flt_ano0 = rmv_runavg(target_ano, window_size=120)  # remove the previous 120-day running averages
    flt_ano = rmv_runavg(flt_ano0, window_size=40)  # remove the previous 40-day running averages

    flt_ano_sel = flt_ano.sel(time=slice(date_sta, date_end))

    # Assign a name to the DataArray
    data_array = flt_ano_sel.rename(varn)

    # Convert the DataArray to a Dataset
    dataset = data_array.to_dataset()

    # Save the Dataset to an NC file
    dataset.to_netcdf(data_store + varn +'.fltano120_40.' + year_sta + 'to' + year_end + 'based' + str(yearclim_sta) + 'to' + str(yearclim_end) + '.nc', mode='w')

    # Close the Dataset
    dataset.close()
    ds.close()


pool = multiprocessing.Pool(19)

pool.starmap(compute_parallel, zip(varn_list, data_path_list))

# Close the pool of worker processes
pool.close()
pool.join()

exit()