'''
This file is to convert raw data to daily data, with the unit of olr converted to W/m2
'''

from util.MJO_indices_util import rawtodaily_1var
import multiprocessing

# variables used 
varn_list = ['olr','prep','tcwv',
    'u200', 'u500','u850',
    'v200','v500','v850',
    'q200','q500','q850',
    'T200','T500','T850',
    'Z200','Z500','Z850',
    'sst']  

# variables used 
varn_name_list = ['ttr','tp','tcwv',
    'u', 'u','u',
    'v','v','v',
    'q','q','q',
    't','t','t',
    'z','z','z',
    'sst'] 
data_path = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/raw/'
data_store = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/daily/'
date_sta = '1978-01-01'
date_end = '2023-12-31'

def convert_parallel(varn, varn_name):
    rawtodaily_1var(varn, varn_name, data_path, data_store, date_sta, date_end)

pool = multiprocessing.Pool(19)

pool.starmap(convert_parallel, zip(varn_list, varn_name_list))

# Close the pool of worker processes
pool.close()
pool.join()