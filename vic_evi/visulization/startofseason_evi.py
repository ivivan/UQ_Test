from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import os
import argparse
import time
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
from matplotlib.dates import drange
import matplotlib.dates as mdates
import plotly.graph_objects as go
import geopandas as gpd
import seaborn as sns
# plt.style.use('seaborn')

sns.color_palette("tab10")
pd.options.display.width = 0

# reproduce
SEED = 15
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



# read csv folder
# read csv files
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
        dfs.append(pd.read_csv(f))

    return dfs, csvs






if __name__ == "__main__":
    # DataLoader definition
    # model hyperparameters

    datadir = "R:/CROPPHEN-Q2067"  #local
    logdir = "./vic_evi/" # local

    ## EVI
    x_path = f'{logdir}/data/all_2020_x.csv'
    y_path = f'{logdir}/data/all_2020_y.csv'


    X = pd.read_csv(x_path,header=0)
    y = pd.read_csv(y_path,header=0)


    # # remove below 0
    # mask_2018 = (X >= 0).all(axis=1)
    # X = X[mask_2018]
    # y = y[mask_2018]




    pixel_field_crop = f'{logdir}/data/all_2020_pixel_field_crop.csv'
    y_Crop_18 = pd.read_csv(pixel_field_crop,header=0)
    y_Crop_18.drop_duplicates(subset=['pixelID'],inplace=True)
    y_Crop_18.dropna(inplace=True)


    folder_path = f'{datadir}/Data/DeepLearningTestData/VIC_EVI/2020_curve_params/'
    alldata, allpaths = data_together(folder_path)

    all_date = pd.concat(alldata)
    all_date = all_date[['pixelID','sos']]
    all_date.drop_duplicates(subset=['pixelID'],inplace=True)
    all_date.dropna(inplace=True)


    final_all = pd.merge(y_Crop_18, all_date,  how='left', left_on=[
        'pixelID'], right_on=['pixelID'])


    final_all2 = pd.concat([final_all,X],axis=1)


    # remove below 0
    mask_2018 = (X >= 0).all(axis=1)
    X = X[mask_2018]

    final_all2 = final_all2[mask_2018]

    final_all2.replace('Chick Pea', 'Chickpea',inplace=True)
    
    ### sos 1/6  151 ###
    sos_set = 151   

    chosen_2018_wheat = final_all2.loc[(final_all2['crop'] == 'Wheat') & (final_all2['sos'] < sos_set)]
    chosen_2018_Canola = final_all2.loc[(final_all2['crop'] == 'Canola')& (final_all2['sos'] < sos_set)]
    chosen_2018_Chickpea = final_all2.loc[(final_all2['crop'] == 'Chickpea')& (final_all2['sos'] < sos_set)]
    chosen_2018_Lentils = final_all2.loc[(final_all2['crop'] == 'Lentils')& (final_all2['sos'] < sos_set)]
    chosen_2018_Barley = final_all2.loc[(final_all2['crop'] == 'Barley')& (final_all2['sos'] < sos_set)]



    ### mean Wheat ###
    mean_valuse = chosen_2018_wheat.mean(axis = 0, skipna = True)
    evi_mean_2018_w =mean_valuse[3:]


    ### mean Canola ###
    mean_valuse = chosen_2018_Canola.mean(axis = 0, skipna = True)
    evi_mean_2018_c = mean_valuse[3:]



    ### mean Chickpea ###
    mean_valuse = chosen_2018_Chickpea.mean(axis = 0, skipna = True)
    evi_mean_2018_cp = mean_valuse[3:]



    ### mean Lentils ###
    mean_valuse = chosen_2018_Lentils.mean(axis = 0, skipna = True)
    evi_mean_2018_l = mean_valuse[3:]



    ### mean Barley ###
    mean_valuse = chosen_2018_Barley.mean(axis = 0, skipna = True)
    evi_mean_2018_b = mean_valuse[3:]




   ##### plot average rain #######

    evi_mean_2018_w = evi_mean_2018_w.to_numpy().tolist()
    evi_mean_2018_c = evi_mean_2018_c.to_numpy().tolist()
    evi_mean_2018_cp = evi_mean_2018_cp.to_numpy().tolist()
    evi_mean_2018_l = evi_mean_2018_l.to_numpy().tolist()
    evi_mean_2018_b = evi_mean_2018_b.to_numpy().tolist()


    base = datetime.date(2020, 4, 1)
    end  = datetime.date(2020, 12, 31)
    delta = datetime.timedelta(days=5)
    l = drange(base, end, delta)

    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.plot(l, evi_mean_2018_w, label='Wheat')
    rects2 = ax.plot(l, evi_mean_2018_c, label='Canola')
    rects3 = ax.plot(l, evi_mean_2018_l, label='Lentils')
    rects4 = ax.plot(l, evi_mean_2018_b, label='Barley')
    rects5 = ax.plot(l, evi_mean_2018_cp, label='Chickpea')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average EVI',fontsize=36)
    ax.set_xlim(base, end)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=22)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    plt.show()
















   