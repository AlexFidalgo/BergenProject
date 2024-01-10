import cdsapi
import zipfile
import os

zip_file_path = 'Bergen_Metrics/download.zip'
extracted_folder_path = 'Bergen_Metrics/download'

if not os.path.exists(extracted_folder_path):
    os.makedirs(extracted_folder_path)

c = cdsapi.Client()

c.retrieve(
    'insitu-gridded-observations-europe',
    {
        'format': 'zip',
        'variable': 'precipitation_amount',
        'product_type': 'ensemble_mean',
        'grid_resolution': '0.25deg',
        'period': '2011_2019',
        'version': '21.0e',
    },
    zip_file_path)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

os.remove(zip_file_path)

print(f"Data extracted to: {extracted_folder_path}")
print(f"{zip_file_path} deleted.")
