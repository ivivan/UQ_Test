from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from catalyst.utils import set_global_seed
from catalyst.dl import SupervisedRunner, Runner
from catalyst.utils import metrics
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback, CriterionCallback, OptunaCallback

import torch
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision

from collections import OrderedDict
import os
import argparse
import time
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# reproduce
SEED = 15
set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    num_classes = 5
    num_gpu = 1
    datadir = "R:/CROPPHEN-Q2067"  #local
    # datadir = "/afm02/Q2/Q2067"  #hpc
    # logdir = "/clusterdata/uqyzha77/Log/vic/winter/" # hpc
    logdir = "./vic_evi/" # local

    ##### vic data
    folder_path = f'{datadir}/Data/DeepLearningTestData/VIC_EVI/2019_cleaned/'
    alldata, allpaths = data_together(folder_path)

    vic2018 = pd.concat(alldata)

    print(vic2018.shape)

    # vic2018['field_id'] = vic2018['field_id'].str.slice(1,-1).str.split('_').str[1]
    vic2018['field_id'] = vic2018['field_id'].str.split('_').str[1]
    vic2018.dropna(inplace=True)

    #### join geo info
    geo_path = f'{datadir}/Data/DeepLearningTestData/VIC_EVI/2019_lat_lon/VIC_latlon_2019.txt'
    geo_df = pd.read_csv(geo_path, sep=',')

    print(vic2018.head())
    print(vic2018.tail())
    print(geo_df.head())

    combined = vic2018.merge(geo_df, on='pixelID')

    print(combined.head())

    labels = combined.iloc[:,1].copy()

    # print(vic2018[vic2018.isnull().any(axis=1)])

    columns_name = list(range(0,55))
    df2 = pd.DataFrame(combined['5d_EVI_gpr'].str.slice(1,-1).str.split(',').values.tolist(),columns=columns_name,dtype=float)
    X = df2
    y = labels
    df_geo = combined[['pixelID','lat','lon']]

    df_geo.to_csv(f"{logdir}/data/all_2019_geo.csv", index=False)



    # X.to_csv(f"{logdir}/data/all_2018_x.csv", index=False)

    # df_test_y = pd.DataFrame(y)

    # df_test_y.to_csv(
    #     f"{logdir}/data/all_2018_y.csv", index=False)





