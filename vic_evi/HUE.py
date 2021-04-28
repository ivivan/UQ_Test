from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from catalyst.utils import set_global_seed
from catalyst.dl import SupervisedRunner, Runner
from catalyst.utils import metrics
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback, CriterionCallback, OptunaCallback
import optuna
from optuna.integration import CatalystPruningCallback

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

    datadir = "R:/CROPPHEN-Q2067"  #local
    # datadir = "/afm02/Q2/Q2067"  #hpc
    # logdir = "/clusterdata/uqyzha77/Log/vic/winter/" # hpc
    logdir = "./vic_evi/" # local


    folder_path = f'{datadir}/Data/DeepLearningTestData/VIC_EVI/HUE/2020/'
    alldata, allpaths = data_together(folder_path)






    ###### for 2018 2019 #########
    vic2018 = pd.concat(alldata)

    columns_name = list(range(0,57))
    df2 = pd.DataFrame(vic2018['VI_site,VI_type, eval_5d_VI_SP'].str.slice(2,-2).str.split(',').values.tolist(),columns=columns_name,dtype=float)

    df2.columns = df2.columns.astype(str)

    df2['56'] = df2['56'].str.slice(0,-2).str[1:]
    df2['2'] = df2['2'].str.slice(8,-1).str[0:]
    df2['0'] = df2['0'].astype(int)

    df2.drop_duplicates(subset=['0'],inplace=True)
    df2.dropna(inplace=True)

    df2.drop('1', axis=1,inplace=True)


    # pixel id
    geo_path = f'{logdir}/data/all_2020_geo.csv'
    geo_df = pd.read_csv(geo_path)

    print('geo_df: {}'.format(geo_df.shape))

    final = pd.merge(geo_df, df2,  how='left', left_on=[
                     'pixelID'], right_on=['0'])

    print('final: {}'.format(final.shape))

    print(final.shape)

    final.drop('pixelID', axis=1,inplace=True)
    final.drop('lat', axis=1,inplace=True)
    final.drop('lon', axis=1,inplace=True)
    final.drop('0', axis=1,inplace=True)

    print(final.shape)
    # print(final.head(40))




    
    # final.iloc[:3000,:].to_csv(f"{logdir}/data/all_2020_hue_sample.csv", index=False)















   