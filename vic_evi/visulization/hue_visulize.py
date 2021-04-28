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
    INPUT_DIM = 1
    OUTPUT_DIM = 5
    HID_DIM = 256
    DROPOUT = 0.2
    RECURRENT_Layers = 2
    LR = 0.004  # learning rate
    EPOCHS = 400
    BATCH_SIZE = 128
    NUM_CLASSES = 5
    num_gpu = 1
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

    ## DegreeDays
    y_DegreeD_path_18 = f'{datadir}/data/all_2018_y_accumulatedD_short.csv'
    y_DegreeD_path_19 = f'{datadir}/data/all_2019_y_accumulatedD_short.csv'
    y_DegreeD_path_20 = f'{datadir}/data/all_2020_y_accumulatedD_short.csv'

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
    y_Crop_path_18 = f'{datadir}/data/all_2018_y.csv'
    y_Crop_path_19 = f'{datadir}/data/all_2019_y.csv'
    y_Crop_path_20 = f'{datadir}/data/all_2020_y.csv'

    y_Crop_18 = pd.read_csv(y_Crop_path_18,header=0)
    y_Crop_19 = pd.read_csv(y_Crop_path_19,header=0)
    y_Crop_20 = pd.read_csv(y_Crop_path_20,header=0)

    y_Crop_18 = y_Crop_18[mask_2018]
    y_Crop_19 = y_Crop_19[mask_2019]
    y_Crop_20 = y_Crop_20[mask_2020]


    ## hue
    y_hue_path_18 = f'{datadir}/data/all_2018_hue.csv'
    y_hue_path_19 = f'{datadir}/data/all_2019_hue.csv'
    y_hue_path_20 = f'{datadir}/data/all_2020_hue.csv'

    y_hue_18 = pd.read_csv(y_hue_path_18,header=0)
    y_hue_19 = pd.read_csv(y_hue_path_19,header=0)
    y_hue_20 = pd.read_csv(y_hue_path_20,header=0)

    y_hue_18 = y_hue_18[mask_2018]
    y_hue_19 = y_hue_19[mask_2019]
    y_hue_20 = y_hue_20[mask_2020]


    # remove below 0
    mask_hue_2018 = ((y_hue_18 <= 1) & (y_hue_18 >= -1)).all(axis=1)
    mask_hue_2019 = ((y_hue_19 <= 1) & (y_hue_19 >= -1)).all(axis=1)
    mask_hue_2020 = ((y_hue_20 <= 1) & (y_hue_20 >= -1)).all(axis=1)


    y_hue_18 = y_hue_18[mask_hue_2018]
    y_Crop_18 = y_Crop_18[mask_hue_2018]
    y_Rain_18 = y_Rain_18[mask_hue_2018]

    y_hue_19 = y_hue_19[mask_hue_2019]
    y_Crop_19 = y_Crop_19[mask_hue_2019]
    y_Rain_19 = y_Rain_19[mask_hue_2019]

    y_hue_20 = y_hue_20[mask_hue_2020]
    y_Crop_20 = y_Crop_20[mask_hue_2020]
    y_Rain_20 = y_Rain_20[mask_hue_2020]

    X = X[mask_hue_2018]
    y = y[mask_hue_2018]

    X_2019 = X_2019[mask_hue_2019]
    y_2019 = y_2019[mask_hue_2019]

    X_2020 = X_2020[mask_hue_2020]
    y_2020 = y_2020[mask_hue_2020]



    # print('X:{}'.format(X.shape))
    # print('y_DegreeD_18:{}'.format(y_DegreeD_18.shape))
    # print('y_Rain_18:{}'.format(y_Rain_18.shape))
    # print('y_Crop_18:{}'.format(y_Crop_18.shape))

    ######## concat ##############
    # ##### 2018
    # y_DegreeD_18_columns_name = list(range(100,155))
    # y_Rain_18_columns_name = list(range(200,255))

    # y_DegreeD_18.columns = y_DegreeD_18_columns_name
    # y_Rain_18.columns = y_Rain_18_columns_name

    # all_2018 = pd.concat([y_Crop_18,X,y_DegreeD_18,y_Rain_18],axis=1)

    # ####### 2019
    # y_DegreeD_18_columns_name = list(range(100,155))
    # y_Rain_18_columns_name = list(range(200,255))

    # y_DegreeD_19.columns = y_DegreeD_18_columns_name
    # y_Rain_19.columns = y_Rain_18_columns_name

    # all_2018 = pd.concat([y_Crop_19,X_2019,y_DegreeD_19,y_Rain_19],axis=1)


    ####### 2020
    y_DegreeD_18_columns_name = list(range(100,155))
    y_Rain_18_columns_name = list(range(200,255))
    y_hue_18_columns_name = list(range(300,355))

    y_DegreeD_20.columns = y_DegreeD_18_columns_name
    y_Rain_20.columns = y_Rain_18_columns_name
    y_hue_20.columns = y_hue_18_columns_name

    all_2018 = pd.concat([y_Crop_20,X_2020,y_DegreeD_20,y_Rain_20,y_hue_20],axis=1)

    print(all_2018.shape)
    print(all_2018.head())


    ############# check  crop field ###############

    chosen_2018 = all_2018.loc[all_2018['field_id'] == 'Barley']
    print(chosen_2018.head())
    print(chosen_2018.shape)
    ### mean ###
    mean_valuse = chosen_2018.mean(axis = 0, skipna = True)
    evi_mean = mean_valuse[0:55]
    deg_mean = mean_valuse[55:110]
    rain_mean = mean_valuse[110:165]
    hue_mean = mean_valuse[165:220]
    print(evi_mean)
    print(deg_mean)
    print(rain_mean)
    print(hue_mean)
    # ### std ###
    # print(chosen_2018.std(axis = 0, skipna = True))


    chosen_2018_wheat = all_2018.loc[all_2018['field_id'] == 'Wheat']
    chosen_2018_Canola = all_2018.loc[all_2018['field_id'] == 'Canola']
    chosen_2018_Chickpea = all_2018.loc[all_2018['field_id'] == 'Chickpea']
    chosen_2018_Lentils = all_2018.loc[all_2018['field_id'] == 'Lentils']
    chosen_2018_Barley = all_2018.loc[all_2018['field_id'] == 'Barley']


    ### mean Wheat ###
    mean_valuse = chosen_2018_wheat.mean(axis = 0, skipna = True)
    evi_mean_2018_w = mean_valuse[0:55]
    deg_mean_2018_w = mean_valuse[55:110]
    rain_mean_2018_w = mean_valuse[110:165]
    hue_mean_2018_w= mean_valuse[165:220]

    ### mean Canola ###
    mean_valuse = chosen_2018_Canola.mean(axis = 0, skipna = True)
    evi_mean_2018_c = mean_valuse[0:55]
    deg_mean_2018_c = mean_valuse[55:110]
    rain_mean_2018_c = mean_valuse[110:165]
    hue_mean_2018_c= mean_valuse[165:220]

    ### mean Chickpea ###
    mean_valuse = chosen_2018_Chickpea.mean(axis = 0, skipna = True)
    evi_mean_2018_cp = mean_valuse[0:55]
    deg_mean_2018_cp = mean_valuse[55:110]
    rain_mean_2018_cp = mean_valuse[110:165]
    hue_mean_2018_cp = mean_valuse[165:220]

    ### mean Lentils ###
    mean_valuse = chosen_2018_Lentils.mean(axis = 0, skipna = True)
    evi_mean_2018_l = mean_valuse[0:55]
    deg_mean_2018_l = mean_valuse[55:110]
    rain_mean_2018_l = mean_valuse[110:165]
    hue_mean_2018_l = mean_valuse[165:220]

    ### mean Barley ###
    mean_valuse = chosen_2018_Barley.mean(axis = 0, skipna = True)
    evi_mean_2018_b = mean_valuse[0:55]
    deg_mean_2018_b = mean_valuse[55:110]
    rain_mean_2018_b = mean_valuse[110:165]
    hue_mean_2018_b = mean_valuse[165:220]


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
    rects1 = ax.plot(l, hue_mean_2018_w, label='Wheat')
    rects2 = ax.plot(l, hue_mean_2018_c, label='Canola')
    rects3 = ax.plot(l, hue_mean_2018_cp, label='Chickpea')
    rects4 = ax.plot(l, hue_mean_2018_l, label='Lentils')
    rects5 = ax.plot(l, hue_mean_2018_b, label='Barley')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average HUE',fontsize=36)
    ax.set_xlim(base, end)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=22)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    plt.show()




    # #### option 2

    # def make_patch_spines_invisible(ax):
    #     ax.set_frame_on(True)
    #     ax.patch.set_visible(False)
    #     for sp in ax.spines.values():
    #         sp.set_visible(False)


    # fig, host = plt.subplots()
    # fig.subplots_adjust(right=0.75)

    # par1 = host.twinx()
    # par2 = host.twinx()

    # # Offset the right spine of par2.  The ticks and label have already been
    # # placed on the right by twinx above.
    # par2.spines["right"].set_position(("axes", 1.1))
    # # Having been created by twinx, par2 has its frame off, so the line of its
    # # detached spine is invisible.  First, activate the frame but make the patch
    # # and spines invisible.
    # make_patch_spines_invisible(par2)
    # # Second, show the right spine.
    # par2.spines["right"].set_visible(True)

    # # p1, = host.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
    # # p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
    # # p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")

    # p1, = host.step(evi_mean.index, hue_mean,"b-",where="post",label="HUE")
    # p2, = par1.step(evi_mean.index, rain_mean,"r-",where="post",label="Accumulated Rainfall")
    # p3, = par2.plot(evi_mean.index, evi_mean,"g-",label="EVI")

    # # host.set_xlim(0, 2)
    # host.set_ylim(0, 4000)
    # par1.set_ylim(0, 800)
    # par2.set_ylim(0, 1)

    # host.set_xlabel("Date",fontsize=36)
    # host.set_ylabel("Accumulated DegreeDays",fontsize=36)
    # par1.set_ylabel("Accumulated Rainfall",fontsize=36)
    # par2.set_ylabel("EVI",fontsize=36)

    # host.yaxis.label.set_color(p1.get_color())
    # par1.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())

    # tkw = dict(size=4, width=1.5)
    # host.tick_params(axis='y',labelsize=16, colors=p1.get_color(), **tkw)
    # par1.tick_params(axis='y',labelsize=16, colors=p2.get_color(), **tkw)
    # par2.tick_params(axis='y',labelsize=16, colors=p3.get_color(), **tkw)
    # host.tick_params(axis='x', **tkw)

    # lines = [p1, p2, p3]

    # host.legend(lines, [l.get_label() for l in lines],fontsize=20,loc="upper left")
    # # plt.savefig("2018-2421-wheat.png", format="png")
    # plt.show()



