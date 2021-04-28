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

    # datadir = "R:/CROPPHEN-Q2067"  #local
    datadir = "./vic_evi/"  #local
    # datadir = "/afm02/Q2/Q2067"  #hpc
    # logdir = "/clusterdata/uqyzha77/Log/vic/winter/" # hpc
    logdir = "./vic_evi/" # local



    # folder_path = f'{datadir}/Data/DeepLearningTestData/VIC_EVI/2020_further_cleaned/'
    folder_path = f'{datadir}/data/partition_data_ori/2020/'
    alldata, allpaths = data_together(folder_path)



    vic_partition = alldata[0]
    vic_geo = alldata[1]
    vic_hue = alldata[2]
    vic_x = alldata[3]
    vic_y = alldata[4]

    ################# pick up zoon ############

    chosen_zoon = vic_partition.loc[(vic_partition['GRDCzones'] == 'Wimmera and Central Vic')]
    picked_pixel = vic_geo.merge(chosen_zoon, on='pixelID',how='left')
    m = (picked_pixel.GRDCzones == 'Wimmera and Central Vic')

    picked_x = vic_x[m]
    picked_y = vic_y[m]
    picked_hue = vic_hue[m]

    picked_x.to_csv(f"{logdir}/data/partition_data_after/2020/all_2020_x_partition.csv", index=False)
    picked_y.to_csv(f"{logdir}/data/partition_data_after/2020/all_2020_y_partition.csv", index=False)
    picked_hue.to_csv(f"{logdir}/data/partition_data_after/2020/all_2020_hue_partition.csv", index=False)
















   