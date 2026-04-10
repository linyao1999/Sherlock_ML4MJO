# This script is used to download data from ERA5 using python

import cdsapi
import os 

def download_1var_single_level(varn, year_sta, year_end, month_list='None', data_path='/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/new_raw/'):
    c = cdsapi.Client()
    year_list = range(year_sta, year_end+1)

    if month_list=='None':
        month_list = [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
        ]

    single_level_var_name = {
        "olr" : "top_net_thermal_radiation",
        "prep" : "total_precipitation",
        "sst" : "sea_surface_temperature",
        "tcwv" : "total_column_water_vapour",
    }

    variable = single_level_var_name[varn]

    print('download ' + variable)


    file_path = f'{data_path}{varn}_2deg_{year_sta}to{year_end}.nc'

    if os.path.exists(file_path):
        print(f"File '{file_path}' exits.")
        return

    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'year': [f'{i}' for i in year_list],
            'month': month_list,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'grid':[
                '2', '2'
            ],
        },
        file_path)


def download_1var_pressure_level(varn, year_sta, year_end, month_list='None', data_path='/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/new_raw/'):
    c = cdsapi.Client()
    year_list = range(year_sta, year_end+1)

    if month_list=='None':
        month_list = [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ]

    varn_short = varn[0]  
    pressure = varn[1:]

    pressure_level_var_name = {
        "q" : "specific_humidity",
        "T" : "temperature",
        "u" : "u_component_of_wind",
        "v" : "v_component_of_wind",
        "Z" : "geopotential",
    }

    variable = pressure_level_var_name[varn_short]

    print('download ' + variable + ' at ' + pressure + ' hPa')


    file_path = f'{data_path}{varn}_2deg_{year_sta}to{year_end}.nc'

    if os.path.exists(file_path):
        print(f"File '{file_path}' exists.")
        return

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'pressure_level': pressure,
            'year': [f'{i}' for i in year_list],
            'month': month_list,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'grid':[
                '2', '2'
            ],
        },
        file_path)


def download_1var(varn, year_sta, year_end, month_list='None', data_path='/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/new_raw/'):
    varn19_p = [
        "u200", "u500", "u850",  
        "q200", "q500", "q850", 
        "T200", "T500", "T850", 
        "Z200", "Z500", "Z850", 
        "v200", "v500", "v850", 
    ]

    varn19_s = [
        "olr", "tcwv", "sst", "prep",
    ]

    if varn in varn19_s: 
        download_1var_single_level(varn, year_sta, year_end, month_list, data_path)
    elif varn in varn19_p:
        download_1var_pressure_level(varn, year_sta, year_end, month_list, data_path)
    else:
        print('Wrong variable! Exit!')


