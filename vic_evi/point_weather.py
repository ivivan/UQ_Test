import urllib.request
from collections import OrderedDict
import os
import argparse
import time
import numpy as np
import pandas as pd
from geopy import distance
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


if __name__ == "__main__":
    logdir = "./vic_evi/"  # local

    lat = -27.50
    lon = 135.00
    start_date = '20180101'
    finish_data = '20181231'
    email = 'yifan.zhang@uq.edu.au'

    r = urllib.request.urlopen(
        # f'https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php?lat={lat}&lon={lon}&start={start_date}&finish={finish_data}&format=csv&username={email}')
        f'https://www.longpaddock.qld.gov.au/cgi-bin/silo/PatchedPointDataset.php?lat={lat}&lon={lon}&start={start_date}&finish={finish_data}&format=csv&username=john.doe@xyz.com.au')
    data = r.read()

    # print(data)

    with open(f"{logdir}/data/sample.txt","wb") as f:
        f.write(data)

    # # weather stations
    # all_stations = f'{logdir}/data/all_stations_vic.csv'
    # stations_df = pd.read_csv(all_stations)
    # stations_df.columns =['id', 'lat', 'lon', 'loc'] 
    # vic_df = stations_df.loc[stations_df['loc'] == 'VIC ']
    # print(vic_df)


    # #### check stations
    # geo_path = f'{logdir}/data/all_2018_geo.csv'
    # geo_df = pd.read_csv(geo_path)

    # ########### field location ###########
    # lat_max = geo_df['lat'].max()
    # lat_min = geo_df['lat'].min()
    # lon_max = geo_df['lon'].max()
    # lon_min = geo_df['lon'].min()

    # lat_diff = (lat_max-lat_min)/3
    # lon_diff = (lon_max-lon_min)/3
    # ######### pick up stations ############

    # chosen_df = vic_df.loc[(vic_df['lat'] >= (lat_min-lat_diff)) & (vic_df['lat'] <= (lat_max+lat_diff))&(vic_df['lon'] <= (lon_max+lon_diff))&(vic_df['lon'] >= (lon_min-lon_diff))]
    



    ############# calculate distance ###############

    









# newport_ri= (41.49008, -71.312796)
# cleveland_oh= (41.499498, -81.695391)
# print(distance.distance(newport_ri, cleveland_oh).miles)
