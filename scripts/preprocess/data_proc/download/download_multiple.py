import os 
import time
from util_multiple import download_1var

# ############ parameters to be changed. ###############
year_sta = 2024
# year_mid = 2000  # if the time series is too long, split it into two parts.
year_end = 2025

month_list = [
    '01', '02', '03',
    '04', '05', '06',
    '07', '08', '09',
    '10', '11', '12',
]

# month_list = [
#     '01', '02', '03',
# ]

# remember to create the following path. 
data_path='/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/new_raw/'

# check if the path exists.
os.makedirs(data_path, exist_ok=True)

# ########################################################
# Do not change below. 
start_time = time.time()
varn = os.environ["varn"]
download_1var(varn, year_sta, year_end, month_list, data_path)
# download_1var(varn, year_sta, year_mid, month_list, data_path)
# download_1var(varn, year_mid+1, year_end, month_list, data_path)

# Calculate elapsed time
elapsed_time = time.time() - start_time

print(f"Download of {varn} complete!")
print(f"Total time elapsed: {elapsed_time:.2f} seconds")

# import cdsapi

# dataset = "reanalysis-era5-pressure-levels"
# request = {
#     "product_type": ["reanalysis"],
#     "variable": ["u_component_of_wind"],
#     "year": ["2022", "2023"],
#     "month": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12"
#     ],
#     "day": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12",
#         "13", "14", "15",
#         "16", "17", "18",
#         "19", "20", "21",
#         "22", "23", "24",
#         "25", "26", "27",
#         "28", "29", "30",
#         "31"
#     ],
#     "time": ["00:00", "06:00"],
#     "pressure_level": ["850"],
#     "data_format": "netcdf",
#     "download_format": "unarchived"
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()
