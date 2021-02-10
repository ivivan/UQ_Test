from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import os
import argparse
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd
import seaborn as sns

sns.color_palette("tab10")
np.set_printoptions(threshold=np.inf)
pd.options.display.width = 0

# reproduce
SEED = 15



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
    datadir = "./vic_evi"  #local
    # datadir = "/afm02/Q2/Q2067/Data/DeepLearningTestData/VIC_EVI"  #hpc
    # logdir = "/afm02/Q2/Q2067/Data/DeepLearningTestData/HPC/Log/vic/1819" # hpc
    logdir = "./vic_evi/" # local

    ## EVI
    x_path = f'{datadir}/data/all_2018_x.csv'
    y_path = f'{datadir}/data/all_2018_y.csv'
    x_2019_path = f'{datadir}/data/all_2019_x.csv'
    y_2019_path = f'{datadir}/data/all_2019_y.csv'
    x_2020_path = f'{datadir}/data/all_2020_x.csv'
    y_2020_path = f'{datadir}/data/all_2020_y.csv'

    X = pd.read_csv(x_path,header=0)
    y = pd.read_csv(y_path,header=0)
    X_2019 = pd.read_csv(x_2019_path,header=0)
    y_2019 = pd.read_csv(y_2019_path,header=0)
    X_2020 = pd.read_csv(x_2020_path,header=0)
    y_2020 = pd.read_csv(y_2020_path,header=0)

    # remove below 0
    mask_2018 = (X >= 0).all(axis=1)
    X = X[mask_2018]
    y = y[mask_2018]

    mask_2019 = (X_2019 >= 0).all(axis=1)
    X_2019 = X_2019[mask_2019]
    y_2019 = y_2019[mask_2019]

    mask_2020 = (X_2020 >= 0).all(axis=1)
    X_2020 = X_2020[mask_2020]
    y_2020 = y_2020[mask_2020]


    Peak_date_2018 = X.idxmax(axis=1).to_numpy(dtype=np.int16)
    Peak_date_2019 = X_2019.idxmax(axis=1).to_numpy(dtype=np.int16)
    Peak_date_2020 = X_2020.idxmax(axis=1).to_numpy(dtype=np.int16)


    # fig, ax = plt.subplots()
    # counts, bins, patches = ax.hist(Peak_date_2020, bins = [0,10,20,30,40,50,60]) 
    # for n, b in zip(counts, bins):
    #     if n > 0:
    #         plt.gca().text(b + 5, n, str(n))  # +0.1 to center text
    # ax.set_xticks(range(0,60,10))
    # # ax.set_title("EVI Peak Date Histgram")
    # ax.set_ylabel('Frequency',fontsize=36)
    # # ax.set_xticklabels('Date')
    # ax.tick_params(labelsize=16)
    # plt.show()

    # #################### peak evi ################

    # fig, ax = plt.subplots()
    # # counts, bins, patches = ax.hist(Peak_date_2018, bins = [0,5,10,15,20,25,30,35,40,45,50,55,60])
    # counts, bins, patches = ax.hist(Peak_date_2019, bins = [0,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]) 
    # plt.axvline(30, color='k', linestyle='dashed', linewidth=1)
    # # for n, b in zip(counts, bins):
    # #     if n > 0:
    # #         plt.gca().text(b + 2, n, str(n))  # +0.1 to center text
    # ax.set_xticks(range(0,60,5))
    # # ax.set_title("EVI Peak Date Histgram")
    # ax.set_ylabel('Frequency',fontsize=36)
    # ax.tick_params(labelsize=16)
    # plt.show()   









    ## DegreeDays
    y_DegreeD_path_18 = f'{datadir}/data/all_2018_y_accumulatedD_short.csv'
    y_DegreeD_path_19 = f'{datadir}/data/all_2019_y_accumulatedD_short.csv'
    y_DegreeD_path_20 = f'{datadir}/data/all_2020_y_accumulatedD.csv'

    y_DegreeD_18 = pd.read_csv(y_DegreeD_path_18,header=0)
    y_DegreeD_19 = pd.read_csv(y_DegreeD_path_19,header=0)
    y_DegreeD_20 = pd.read_csv(y_DegreeD_path_20,header=0)

    y_DegreeD_18 = y_DegreeD_18[mask_2018]
    y_DegreeD_19 = y_DegreeD_19[mask_2019]
    y_DegreeD_20 = y_DegreeD_20[mask_2020]

    ## Accumulated Rain
    y_Rain_path_18 = f'{datadir}/data/all_2018_y_accumulated_rain_final.csv'
    y_Rain_path_19 = f'{datadir}/data/all_2019_y_accumulated_rain_final.csv'
    y_Rain_path_20 = f'{datadir}/data/all_2020_y_accumulated_rain_final.csv'

    y_Rain_18 = pd.read_csv(y_Rain_path_18,header=0)
    y_Rain_19 = pd.read_csv(y_Rain_path_19,header=0)
    y_Rain_20 = pd.read_csv(y_Rain_path_20,header=0)

    y_Rain_18 = y_Rain_18[mask_2018]
    y_Rain_19 = y_Rain_19[mask_2019]
    y_Rain_20 = y_Rain_20[mask_2020]

    ## crop field
    y_Crop_path_18 = f'{datadir}/data/all_2018_pixel_field_crop.csv'
    y_Crop_path_19 = f'{datadir}/data/all_2019_pixel_field_crop.csv'
    y_Crop_path_20 = f'{datadir}/data/all_2020_pixel_field_crop.csv'

    y_Crop_18 = pd.read_csv(y_Crop_path_18,header=0)
    y_Crop_19 = pd.read_csv(y_Crop_path_19,header=0)
    y_Crop_20 = pd.read_csv(y_Crop_path_20,header=0)

    y_Crop_18 = y_Crop_18[mask_2018]
    y_Crop_19 = y_Crop_19[mask_2019]
    y_Crop_20 = y_Crop_20[mask_2020]


    # print('X:{}'.format(X.shape))
    # print('y_DegreeD_18:{}'.format(y_DegreeD_18.shape))
    # print('y_Rain_18:{}'.format(y_Rain_18.shape))
    # print('y_Crop_18:{}'.format(y_Crop_18.shape))

    ######## concat ##############
    ##### 2018
    y_DegreeD_18_columns_name = list(range(100,155))
    y_Rain_18_columns_name = list(range(200,255))

    y_DegreeD_18.columns = y_DegreeD_18_columns_name
    y_Rain_18.columns = y_Rain_18_columns_name

    all_2018 = pd.concat([y_Crop_18,X,y_DegreeD_18,y_Rain_18],axis=1)

    ####### 2019
    y_DegreeD_18_columns_name = list(range(100,155))
    y_Rain_18_columns_name = list(range(200,255))

    y_DegreeD_19.columns = y_DegreeD_18_columns_name
    y_Rain_19.columns = y_Rain_18_columns_name

    all_2019 = pd.concat([y_Crop_19,X_2019,y_DegreeD_19,y_Rain_19],axis=1)


    ####### 2020
    y_DegreeD_18_columns_name = list(range(100,155))
    y_Rain_18_columns_name = list(range(200,255))

    y_DegreeD_20.columns = y_DegreeD_18_columns_name
    y_Rain_20.columns = y_Rain_18_columns_name

    all_2020 = pd.concat([y_Crop_20,X_2020,y_DegreeD_20,y_Rain_20],axis=1)


    # ############# compare rain ###############

    # # # chosen_2018 = all_2018.loc[(all_2018['field'] == 36451) & (all_2018['crop'] == 'Barley')]
    # # chosen_2018 = all_2018.loc[all_2018['crop'] == 'Barley']
    # # print(chosen_2018.head())
    # # print(chosen_2018.shape)
    # ### mean 2018 ###
    # mean_valuse = all_2018.mean(axis = 0, skipna = True)
    # evi_mean_2018 = mean_valuse[2:57]
    # deg_mean_2018 = mean_valuse[57:112]
    # rain_mean_2018 = mean_valuse[112:167]

    # ### mean 2019 ###
    # mean_valuse = all_2019.mean(axis = 0, skipna = True)
    # evi_mean_2019 = mean_valuse[2:57]
    # deg_mean_2019 = mean_valuse[57:112]
    # rain_mean_2019 = mean_valuse[112:167]

    # ### mean 2020 ###
    # mean_valuse = all_2020.mean(axis = 0, skipna = True)
    # evi_mean_2020 = mean_valuse[2:57]
    # deg_mean_2020 = mean_valuse[57:112]
    # rain_mean_2020 = mean_valuse[112:167]


    # ##### plot average rain #######

    # rain_mean_2018 = rain_mean_2018.to_numpy().tolist()
    # rain_mean_2019 = rain_mean_2019.to_numpy().tolist()
    # rain_mean_2020 = rain_mean_2020.to_numpy().tolist()



    # labels = list(range(0,55))


    # x = np.arange(len(labels))  # the label locations
    # width = 0.2  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width/2, rain_mean_2018, width, label='2018')
    # rects2 = ax.bar(x + width/2, rain_mean_2019, width, label='2019')
    # rects3 = ax.bar(x + width/2 + width, rain_mean_2020, width, label='2020')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Accumulated Rainfall (mm)',fontsize=36)
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.tick_params(labelsize=16)
    # ax.legend()

    # fig.tight_layout()

    # plt.show()



    ############# compare dd ###############

    chosen_2018 = all_2018.loc[all_2018['crop'] == 'Wheat']
    chosen_2019 = all_2019.loc[all_2019['crop'] == 'Wheat']
    chosen_2020 = all_2020.loc[all_2020['crop'] == 'Wheat']
    # print(chosen_2018.head())
    # print(chosen_2018.shape)
    ### mean 2018 ###
    mean_valuse = chosen_2018.mean(axis = 0, skipna = True)
    evi_mean_2018 = mean_valuse[2:57]
    deg_mean_2018 = mean_valuse[57:112]
    rain_mean_2018 = mean_valuse[112:167]

    ### mean 2019 ###
    mean_valuse = chosen_2019.mean(axis = 0, skipna = True)
    evi_mean_2019 = mean_valuse[2:57]
    deg_mean_2019 = mean_valuse[57:112]
    rain_mean_2019 = mean_valuse[112:167]

    ### mean 2020 ###
    mean_valuse = chosen_2020.mean(axis = 0, skipna = True)
    evi_mean_2020 = mean_valuse[2:57]
    deg_mean_2020 = mean_valuse[57:112]
    rain_mean_2020 = mean_valuse[112:167]


    ##### plot average rain #######

    deg_mean_2018 = deg_mean_2018.to_numpy().tolist()
    deg_mean_2019 = deg_mean_2019.to_numpy().tolist()
    deg_mean_2020 = deg_mean_2020.to_numpy().tolist()



    labels = list(range(0,55))


    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()

    rects1, = ax.step(labels, deg_mean_2018,"b-",where="post",label="Accumulated DegreeDays 2018")
    rects2, = ax.step(labels, deg_mean_2019,"r-",where="post",label="Accumulated DegreeDays 2019")
    # rects3, = ax.step(labels, deg_mean_2020,"g-",where="post",label="Accumulated DegreeDays 2020")

    # rects1 = ax.bar(x - width/2, deg_mean_2018, width, label='2018')
    # rects2 = ax.bar(x + width/2, deg_mean_2019, width, label='2019')
    # rects3 = ax.bar(x + width/2 + width, deg_mean_2020, width, label='2020')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accumulated DegreeDays',fontsize=36)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=20)

    fig.tight_layout()

    plt.show()


    # ############# compare evi ###############

    # # # chosen_2018 = all_2018.loc[(all_2018['field'] == 36451) & (all_2018['crop'] == 'Barley')]
    # chosen_2018 = all_2018.loc[all_2018['crop'] == 'Wheat']
    # chosen_2019 = all_2019.loc[all_2019['crop'] == 'Wheat']
    # chosen_2020 = all_2020.loc[all_2020['crop'] == 'Wheat']

    # # chosen_2018 = all_2020.loc[all_2020['crop'] == 'Barley']
    # # chosen_2019 = all_2020.loc[all_2020['crop'] == 'Barley']
    # # chosen_2020 = all_2020.loc[all_2020['crop'] == 'Barley']
    # # print(chosen_2018.head())
    # # print(chosen_2018.shape)
    # ### mean 2018 ###
    # mean_valuse = chosen_2018.mean(axis = 0, skipna = True)
    # evi_mean_2018 = mean_valuse[2:57]
    # deg_mean_2018 = mean_valuse[57:112]
    # rain_mean_2018 = mean_valuse[112:167]

    # ### mean 2019 ###
    # mean_valuse = chosen_2019.mean(axis = 0, skipna = True)
    # evi_mean_2019 = mean_valuse[2:57]
    # deg_mean_2019 = mean_valuse[57:112]
    # rain_mean_2019 = mean_valuse[112:167]

    # ### mean 2020 ###
    # mean_valuse = chosen_2020.mean(axis = 0, skipna = True)
    # evi_mean_2020 = mean_valuse[2:57]
    # deg_mean_2020 = mean_valuse[57:112]
    # rain_mean_2020 = mean_valuse[112:167]


    # ##### plot average rain #######

    # evi_mean_2018 = evi_mean_2018.to_numpy().tolist()
    # evi_mean_2019 = evi_mean_2019.to_numpy().tolist()
    # evi_mean_2020 = evi_mean_2020.to_numpy().tolist()



    # labels = list(range(0,55))


    # x = np.arange(len(labels))  # the label locations
    # width = 0.2  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.plot(labels, evi_mean_2018, label='2018')
    # rects2 = ax.plot(labels, evi_mean_2019, label='2019')
    # rects3 = ax.plot(labels, evi_mean_2020, label='2020')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Average EVI',fontsize=36)
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.tick_params(labelsize=16)
    # ax.legend(fontsize=20)
    # ax.set_ylim(0, 1)

    # fig.tight_layout()

    # plt.show()










