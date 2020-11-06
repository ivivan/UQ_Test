from LSTM_Classification_V3 import AttentionModel, clip_gradient
# from LSTM_Classification import AttentionModel, clip_gradient
from loss import LabelSmoothingCrossEntropy
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catalyst.utils import set_global_seed
from catalyst.dl import SupervisedRunner, Runner
from catalyst.utils import metrics
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback, CriterionCallback


import torch
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

from collections import OrderedDict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd
import seaborn as sns
import os
import time
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
pd.options.display.width = 0
sns.color_palette("tab10")

# reproduce
SEED = 15
set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# determine the supported device


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device

# convert a df to tensor to be used in pytorch


def numpy_to_tensor(ay, tp):
    device = get_device()
    return torch.from_numpy(ay).type(tp).to(device)

# count model parameters


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CustomRunner(Runner):

    def _handle_batch(self, batch):
        x, y = batch
        # y_hat, attention = self.model(x)
        outputs = self.model(x)

        loss = F.cross_entropy(outputs['logits'], y)
        accuracy01, accuracy02 = metrics.accuracy(
            outputs['logits'], y, topk=(1, 2))
        self.batch_metrics = {
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy02": accuracy02,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


if __name__ == "__main__":
    # sample data

    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/cleaned_data_25753.csv'
    # df_all = pd.read_csv(data_path)
    # field_id = df_all.iloc[:, 1]
    # df_all = df_all.rename(columns={"filed_id_num": "field_id"})
    # labels = df_all.iloc[:, 2]
    # # check sample no.
    # print('all:',df_all.shape[0])
    # unique_elements, counts_elements = np.unique(labels, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))




    # # # pick up only NDVI,and paddocktyp
    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/parameter_plus_NDVI_53525.csv'
    # df_all = pd.read_csv(data_path)
    # field_id = df_all.iloc[:, 1]
    # df_all = df_all.rename(columns={"filed_id_num": "field_id"})
    # labels = df_all.iloc[:, 2]
    # # check sample no.
    # print('all:',df_all.shape[0])
    # unique_elements, counts_elements = np.unique(labels, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))



    # # # pick up only NDVI,and paddocktyp
    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/VIC_ready2use150000.csv'
    # df_all2 = pd.read_csv(data_path)
    # BeforeSymbol = df_all2['field_id'].str.split('_').str[0]
    # df_all2['field_id'] = BeforeSymbol

    # df_all2 = df_all2.iloc[:, 6:].copy()
    # labels = df_all2.columns[1:]

    # X = df_all2[labels]
    # y = df_all2['paddocktyp']


    # # check sample no.
    # print('all:',df_all2.shape[0])
    # unique_elements, counts_elements = np.unique(y, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))



    ## check field

    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/cleaned_data_25753.csv'
    # df_all = pd.read_csv(data_path)
    # field_id = df_all.iloc[:, 1]
    # df_all = df_all.rename(columns={"filed_id_num": "field_id"})
    # df_picked = df_all[['crop', 'field_id']]
    # df_count = df_picked.groupby('crop').agg({'field_id': 'nunique'})
    # print(df_count)



    # # # pick up only NDVI,and paddocktyp
    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/parameter_plus_NDVI_53525.csv'
    # df_all = pd.read_csv(data_path)
    # field_id = df_all.iloc[:, 1]
    # df_all = df_all.rename(columns={"filed_id_num": "field_id"})

    # # check sample no.
    # df_picked = df_all[['crop', 'field_id']]
    # df_count = df_picked.groupby('crop').agg({'field_id': 'nunique'})
    # print(df_count)


    # # # pick up only NDVI,and paddocktyp
    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/VIC_ready2use150000.csv'
    # df_all = pd.read_csv(data_path)
    # # check sample no.
    # df_picked = df_all[['paddocktyp', 'field_id']]
    # df_count = df_picked.groupby('paddocktyp').agg({'field_id': 'nunique'})
    # print(df_count)


    # # pick up only NDVI,and paddocktyp
    data_path = 'crop_classification_multi/downloadtf/train_x_download.csv'
    df_all = pd.read_csv(data_path)
    print(df_all.head(1))










