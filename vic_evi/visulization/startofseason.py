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
import matplotlib.pyplot as plt
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
    logdir = "./vic_evi/" # local


    folder_path = f'{datadir}/Data/DeepLearningTestData/VIC_EVI/2020_curve_params/'
    alldata, allpaths = data_together(folder_path)

    all_date = pd.concat(alldata)
    all_date = all_date[['pixelID','field_id','sos']]
    all_date.drop_duplicates(subset=['pixelID'],inplace=True)
    all_date.dropna(inplace=True)


    # ######## 2018 ###########
    # all_date['crop'] = all_date['field_id'].str.slice(1,-1).str.split('_').str[1]
    # all_date['field_id'] = all_date['field_id'].str.slice(2,-1).str.split('_').str[0]

    # ######## 2019 ###########
    # all_date['crop'] = all_date['field_id'].str.split('_').str[1]
    # all_date['field_id'] = all_date['field_id'].str.split('_').str[0]


    ######## 2020 ###########
    all_date['crop'] = all_date['field_id'].str.slice(1,-1).str.split('_').str[1]
    all_date['field_id'] = all_date['field_id'].str.slice(2,-1).str.split('_').str[0]

    # print(all_date.head())


    ############# use to choose big or small size field ###############
    results = all_date.groupby(['crop', 'field_id']).describe()['sos'].reset_index()
    chosen_crop =results.loc[(results['crop']=='Barley') & (results['count'] < 200) ]

    print(chosen_crop)
    chosen_crop = chosen_crop[['crop','field_id']]

    unique_field_id = chosen_crop['field_id'].unique()

    chosen_data = all_date.loc[all_date['field_id'].isin(unique_field_id[0:15])]

    green_diamond = dict(markerfacecolor='g', marker='D')
    
    chosen_data.boxplot(column='sos', by='field_id',flierprops=green_diamond,fontsize=20)

    plt.show()







   