import urllib.request
from collections import OrderedDict
import os
import argparse
import time
import numpy as np
import pandas as pd
from geopy import distance

from sklearn.preprocessing import LabelEncoder, StandardScaler

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def data_together(filepath):
    csvs = []
    dfs = []

    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".csv"):
                csvs.append(filepath)

    for f in csvs:
        df = pd.read_csv(f)
        daily_rain = df["daily_rain"].to_numpy()
        max_temp = df["max_temp"].to_numpy()
        min_temp = df["min_temp"].to_numpy()
        station = df.iloc[0, 0]

        d = {'station': station, 'rain':[daily_rain], 'max_temp': [
            max_temp], 'min_temp': [min_temp]}
        new_df = pd.DataFrame(data=d)

        dfs.append(new_df)

    return dfs, csvs




# read csv files
def data_csv(filepath):
    csvs = []
    dfs = []

    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".csv"):
                csvs.append(filepath)

    for f in csvs:
        dfs.append(pd.read_csv(f))

    return dfs, csvs




def myround(x, prec=2, base=.05):
    return round(base * round(float(x)/base), prec)

def myround_five(x, prec=0, base=5):
    return round(base * round(float(x)/base), prec)


def cal_distance(station_lat, station_lon, target_lat, target_lon):
    return distance.distance((station_lat, station_lon), (target_lat, target_lon)).miles


def find_station(target_lat, target_lon, chosen_df):

    chosen_df['dist'] = chosen_df.apply(lambda row: cal_distance(
        row['lat'], row['lon'], target_lat, target_lon), axis=1)

    station_id = chosen_df.iloc[chosen_df['dist'].idxmin(), 0]

    return station_id


def rain_before_sown_list(rain_list, doy):

    return rain_list[0:doy].tolist()


def rain_before_sown_accumulated(rain_list,doy):

    return np.add.accumulate(rain_list[0:doy]).tolist()


if __name__ == "__main__":
    datadir = "R:/CROPPHEN-Q2067"  #local
    logdir = "./vic120/"  # local


    # weather stations
    all_stations = f'{logdir}/data/all_stations_vic.csv'
    stations_df = pd.read_csv(all_stations)
    stations_df.columns = ['id', 'lat', 'lon', 'loc']
    vic_df = stations_df.loc[stations_df['loc'] == 'VIC ']
    vic_df['lat'] = vic_df['lat'].apply(myround)
    vic_df['lon'] = vic_df['lon'].apply(myround)

    # check stations

    vic120_path = f'{logdir}/data/Final2.csv'
    vic120_df = pd.read_csv(vic120_path)

    vic120_df['lat'] = vic120_df['lat'].apply(myround)
    vic120_df['lon'] = vic120_df['lon'].apply(myround)

    unique_geo = vic120_df.groupby(['lat', 'lon']).size().reset_index(name='Freq')


    ########### field location ###########
    lat_max = vic120_df['lat'].max()
    lat_min = vic120_df['lat'].min()
    lon_max = vic120_df['lon'].max()
    lon_min = vic120_df['lon'].min()

    lat_diff = (lat_max-lat_min)/10
    lon_diff = (lon_max-lon_min)/10
    ######### pick up stations ############

    chosen_df = vic_df.loc[(vic_df['lat'] >= (lat_min-lat_diff)) & (vic_df['lat'] <= (lat_max+lat_diff))
                           & (vic_df['lon'] <= (lon_max+lon_diff)) & (vic_df['lon'] >= (lon_min-lon_diff))]

    chosen_df.reset_index(drop=True, inplace=True)


    ############# calculate distance and choose station ###############

    unique_geo['station'] = unique_geo.apply(
        lambda row: find_station(row['lat'], row['lon'], chosen_df), axis=1)

    ############# merge back ################

    final = pd.merge(vic120_df, unique_geo,  how='left', left_on=[
                     'lat', 'lon'], right_on=['lat', 'lon'])

    final.drop(columns=['Freq'], inplace=True)

    near_stations = final['station'].unique()

    # print('final:{}'.format(final.shape))
    print('final:{}'.format(final.head))


    # ############# download data ################

    # for i in near_stations:
    #     r = urllib.request.urlopen(
    #         f'https://www.longpaddock.qld.gov.au/cgi-bin/silo/PatchedPointDataset.php?station={i}&start=20200101&finish=20201231&format=csv&comment=rxn&username=john.doe@xyz.com.au')
    #     data = r.read()
    #     with open(f"{logdir}/data/stations_2020/station_id_{i}.csv", "wb") as f:
    #         f.write(data)



    # ############### temp #################

    folder_path = f'{logdir}/data/stations_2020/'
    alldata, allpaths = data_together(folder_path)

    temp_df = pd.concat(alldata)

    # temp_df = temp_df[['station','max_temp','min_temp']]
    temp_df.reset_index(drop=True, inplace=True)

    print(temp_df.head())



    final_all = pd.merge(final, temp_df,  how='left', left_on=[
        'station'], right_on=['station'])


    print(final_all.head())
    print('final_all:{}'.format(final_all.shape))
    print(final_all[final_all.isnull().any(axis=1)])

    final_all.to_csv(
        f"{logdir}/data/all_information_120_2.csv", index=False)

    # ########### pick up rain days ##############

    final_all['rain_sub'] = final_all.apply(
        lambda row: rain_before_sown_list(row['rain'], row['doy']), axis=1)

    final_all['rain_sub_accumulated'] = final_all.apply(
        lambda row: rain_before_sown_accumulated(row['rain'], row['doy']), axis=1)

    print(final_all.head())
    print('final_all:{}'.format(final_all.shape))
    print(final_all[final_all.isnull().any(axis=1)])

    picked_df = final_all[['Crop','doy','rain_sub','rain_sub_accumulated']]

    print(picked_df)


    picked_df.to_csv(
        f"{logdir}/data/check2.csv", index=False)



