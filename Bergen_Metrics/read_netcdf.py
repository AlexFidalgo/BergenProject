import os
import netCDF4
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
download_folder = os.path.join(script_dir, 'download')
nc_file_path = os.path.join(download_folder, 'tg_ens_mean_0.25deg_reg_2011-2023_v28.0e.nc')

## To check data fields
# with netCDF4.Dataset(nc_file_path, 'r') as nc_file:
    
#     print(f"NetCDF file '{nc_file_path}' opened successfully.")
#     print("File Dimensions:")
#     for dim_name, dim_size in nc_file.dimensions.items():
#         print(f" - {dim_name}: {dim_size.size}")
    
#     print("\nFile Variables:")
#     for var_name, var in nc_file.variables.items():
#         print(f" - {var_name}: {var.shape} ({var.units})")

with netCDF4.Dataset(nc_file_path, 'r') as nc_file:
    latitude = nc_file.variables['latitude'][:]
    longitude = nc_file.variables['longitude'][:]
    tg = nc_file.variables['tg'][:]
    time = nc_file.variables['time'][:]

    rows_to_read = 10
    df = pd.DataFrame(tg[:rows_to_read, :, :].reshape(rows_to_read, -1), columns=pd.MultiIndex.from_product([latitude, longitude], names=['Latitude', 'Longitude']))
    
    df['Time'] = pd.to_datetime("1950-01-01") + pd.to_timedelta(time[:rows_to_read], unit='D')


x = 1
    