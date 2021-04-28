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
    folder_path = f'{datadir}/data/partition/'
    alldata, allpaths = data_together(folder_path)

    vic2018 = alldata[0]
    vic2019 = alldata[1]
    vic2020 = alldata[2]

    print(vic2020.head())

    vic2018.drop_duplicates(subset=['pixelID'],inplace=True)
    vic2018.dropna(inplace=True)

    vic2019.drop_duplicates(subset=['pixID'],inplace=True)
    vic2019.dropna(inplace=True)

    vic2020.drop_duplicates(subset=['pixID'],inplace=True)
    vic2020.dropna(inplace=True)

    vic2018 = vic2018[['pixelID','GRDCzones']]
    vic2019 = vic2019[['pixID','GRDCzones']]
    vic2020 = vic2020[['pixID','GRDCzones']]

    vic2019.columns = ['pixelID', 'GRDCzones']
    vic2020.columns = ['pixelID', 'GRDCzones']

    print(vic2019.head())



    vic2018.to_csv(f"{logdir}/data/partition_data_ori/2018_partation.csv", index=False)
    vic2019.to_csv(f"{logdir}/data/partition_data_ori/2019_partation.csv", index=False)
    vic2020.to_csv(f"{logdir}/data/partition_data_ori/2020_partation.csv", index=False)










   