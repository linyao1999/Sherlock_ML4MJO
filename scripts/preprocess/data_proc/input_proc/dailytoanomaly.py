'''
This file is to compute daily anomaly from daily data.
Anomaly means that we remove the time mean and the first three harmonics of the annual cycle.
=============
input: [time, lat, lon]
output: [time, lat, lon]

'''

# salloc --nodes 1 --qos interactive --time 03:10:00 --constraint cpu --account=m3312


from util.MJO_indices_util import get_anomalies_1var
import multiprocessing
import xarray as xr 

# ========= parameters to be set ================
# we will get data anomalies for periods of date_sta to date_end
date_sta = '1978-01-01'
date_end = '2025-12-31'

# this indicates the time range of daily data file used. 
year_sta = '1978'
year_end = '2025'

# this indicates which years are used to get the annual cycle.
yearclim_sta=1979
yearclim_end=2001

# daily data is stored in this directory.
dirn = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/daily/'

# anomaly output is store in this directory.
# NOTE: remember to create the directory. 
data_store = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/anomaly/'
# ================================================


# =========== do not change codes below ================
# variable used 
varn_list = [
    'olr','prep','tcwv',
    'u200', 'u500','u850',
    'v200','v500','v850',
    'q200','q500','q850',
    'T200','T500','T850',
    'Z200','Z500','Z850',
    'sst',
]

data_path_list = [dirn + vari +'.day.nc' for vari in varn_list]

# define a function to run get_anomalies_1var
def compute_parallel(varn, data_path):
    target_ano = get_anomalies_1var(varn, data_path, date_sta, date_end, latsel=90, yearclim_sta=yearclim_sta, yearclim_end=yearclim_end)

    # Assign a name to the DataArray
    data_array = target_ano.rename(varn)

    # Convert the DataArray to a Dataset
    dataset = data_array.to_dataset()

    # Save the Dataset to an NC file
    dataset.to_netcdf(data_store + varn +'.anomaly.' + year_sta + 'to' + year_end + 'based' + str(yearclim_sta) + 'to' + str(yearclim_end) + '.nc', mode='w')

    # Close the Dataset
    dataset.close()

pool = multiprocessing.Pool(19)

pool.starmap(compute_parallel, zip(varn_list, data_path_list))

# Close the pool of worker processes
pool.close()
pool.join()

