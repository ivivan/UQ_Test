import urllib.request
from collections import OrderedDict
import os
import argparse
import time
import numpy as np
import pandas as pd
from geopy import distance


from degree_days.calculation import calcDegreeDays, degreeDays_list, all_degreeDays
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
        # df = df[90::1]
        daily_rain = df["daily_rain"].to_numpy()
        max_temp = df["max_temp"][90::1].to_numpy()
        min_temp = df["min_temp"][90::1].to_numpy()
        station = df.iloc[0, 0]

        d = {'station': station, 'rain':[daily_rain], 'max_temp': [
            max_temp], 'min_temp': [min_temp]}
        new_df = pd.DataFrame(data=d)

        dfs.append(new_df)

    return dfs, csvs


def myround(x, prec=2, base=.05):
    return round(base * round(float(x)/base), prec)


def cal_distance(station_lat, station_lon, target_lat, target_lon):
    return distance.distance((station_lat, station_lon), (target_lat, target_lon)).miles


def find_station(target_lat, target_lon, chosen_df):

    chosen_df['dist'] = chosen_df.apply(lambda row: cal_distance(
        row['lat'], row['lon'], target_lat, target_lon), axis=1)

    station_id = chosen_df.iloc[chosen_df['dist'].idxmin(), 0]

    return station_id


if __name__ == "__main__":
    logdir = "./vic_evi/"  # local

    # r = urllib.request.urlopen(
    #     'https://www.longpaddock.qld.gov.au/cgi-bin/silo/PatchedPointDataset.php?format=near&station=80052&radius=10000&sortby=name')
    # data = r.read()

    # with open(f"{logdir}/data/all_stations.txt","wb") as f:
    #     f.write(data)

    # weather stations
    all_stations = f'{logdir}/data/all_stations_vic.csv'
    stations_df = pd.read_csv(all_stations)
    stations_df.columns = ['id', 'lat', 'lon', 'loc']
    vic_df = stations_df.loc[stations_df['loc'] == 'VIC ']
    vic_df['lat'] = vic_df['lat'].apply(myround)
    vic_df['lon'] = vic_df['lon'].apply(myround)

    # check stations
    geo_path = f'{logdir}/data/all_2018_geo.csv'
    geo_df = pd.read_csv(geo_path)
    geo_df['lat'] = geo_df['lat'].apply(myround)
    geo_df['lon'] = geo_df['lon'].apply(myround)

    unique_geo = geo_df.groupby(['lat', 'lon']).size().reset_index(name='Freq')

    print(geo_df.shape)
    print(geo_df)

    ########### field location ###########
    lat_max = geo_df['lat'].max()
    lat_min = geo_df['lat'].min()
    lon_max = geo_df['lon'].max()
    lon_min = geo_df['lon'].min()

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

    final = pd.merge(geo_df, unique_geo,  how='left', left_on=[
                     'lat', 'lon'], right_on=['lat', 'lon'])

    final.drop(columns=['Freq'], inplace=True)

    near_stations = final['station'].unique()

    ############# download data ################

    # for i in near_stations:
    #     r = urllib.request.urlopen(
    #         f'https://www.longpaddock.qld.gov.au/cgi-bin/silo/PatchedPointDataset.php?station={i}&start=20190101&finish=20191231&format=csv&comment=rxn&username=john.doe@xyz.com.au')
    #     data = r.read()
    #     with open(f"{logdir}/data/stations_2019/station_id_{i}.csv", "wb") as f:
    #         f.write(data)

    ############### temp #################

    folder_path = f'{logdir}/data/stations_2018/'
    alldata, allpaths = data_together(folder_path)

    temp_df = pd.concat(alldata)

    # temp_df = temp_df[['station','max_temp','min_temp']]
    temp_df.reset_index(drop=True, inplace=True)

    ########### all crop for same temp ############

    all_crop = ['Barley', 'Canola', 'Lentils', 'Wheat', 'Chickpea']

    temp_df['degreeDay'] = temp_df.apply(
        lambda row: all_degreeDays(row['max_temp'], row['min_temp'], all_crop), axis=1)

    ########### add degree day ##############

    # temp_df['degree_day'] = temp_df.apply(
    #     lambda row: calcDegreeDays(row['max_temp'], row['min_temp']), axis=1)

    ########### EVI data ##############

    y_path = f'{logdir}/data/all_2018_y.csv'
    y = pd.read_csv(y_path, header=0)

    crops = y.to_numpy()

    # crops = np.squeeze(crops)

    final_crop = pd.concat([final, y], axis=1)
    final_all = pd.merge(final_crop, temp_df,  how='left', left_on=[
        'station'], right_on=['station'])


    # print(final_all.head())

    ########### save final degreeday data ##############

    # pixel_crop = final_all['field_id']
    # crops_degreeD = final_all['degreeDay']
    # all_degreeD = []

    # for i in range(0, final_all.shape[0]):
    #     crop = pixel_crop[i]
    #     temp = crops_degreeD[i]
    #     degreeD = temp[crop]
    #     all_degreeD.append(degreeD)

    # results = pd.DataFrame({'field_id': pixel_crop, 'degreeDay': all_degreeD})
    # print(results.shape)

    # results.to_csv(
    #     f"{logdir}/data/all_2018_y_degreeD.csv", index=False)



    ########### save final accumulated rainfall data ##############

    pixel_crop = final_all['field_id']
    daily_rain = final_all['rain']
    accumulated_rain = []

    for i in range(0, final_all.shape[0]):
        crop = pixel_crop[i]
        rain_d = daily_rain[i]
        accumulated_rain_d = np.add.accumulate(rain_d)
        accumulated_rain.append(accumulated_rain_d.tolist())

    results = pd.DataFrame({'field_id': pixel_crop, 'accumulated_rain': accumulated_rain})
    print(results.shape)
    print(results.head())

    # results.to_csv(
    #     f"{logdir}/data/all_2018_y_accumulated_rain.csv", index=False)
