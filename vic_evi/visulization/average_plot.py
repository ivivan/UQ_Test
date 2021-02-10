from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
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
import plotly.graph_objects as go
import geopandas as gpd
import seaborn as sns

sns.color_palette("tab10")
np.set_printoptions(threshold=np.inf)
pd.options.display.width = 0

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


# gradient clip
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


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

# classification label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()



# self attention with lstm
class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, recurrent_layers, dropout_p):
        super(AttentionModel, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.recurrent_layers = recurrent_layers
        self.dropout_p = dropout_p

        self.input_embeded = nn.Linear(input_dim, hidden_dim//2)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(input_size=hidden_dim//2, hidden_size=hidden_dim, num_layers=recurrent_layers,
                            bidirectional=True)

        self.self_attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(True),
            nn.Linear(hidden_dim*2, 1)
        )

        self.scale = 1.0/np.sqrt(hidden_dim)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.output_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.label = nn.Linear(hidden_dim*4, output_dim)

    def forward(self, input_sentences):

        input = self.dropout(torch.tanh(self.input_embeded(input_sentences)))
        input = input.permute(1, 0, 2)

        self.lstm.flatten_parameters()

        output, (final_hidden_state, final_cell_state) = self.lstm(
            input)
        output = output.permute(1, 0, 2)

        attn_ene = self.self_attention(output)

        attn_ene = attn_ene.view(
            output.shape[0], -1)
        
        # scale
        attn_ene.mul_(self.scale)

        attns = F.softmax(attn_ene, dim=1).unsqueeze(2)

        final_inputs = (output * attns).sum(dim=1)
        final_inputs2 = output.sum(dim=1)

        combined_inputs = torch.cat([final_inputs, final_inputs2], dim=1)

        logits = self.label(combined_inputs)

        return logits



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
    # ##### 2018
    # y_DegreeD_18_columns_name = list(range(100,155))
    # y_Rain_18_columns_name = list(range(200,255))

    # y_DegreeD_18.columns = y_DegreeD_18_columns_name
    # y_Rain_18.columns = y_Rain_18_columns_name

    # all_2018 = pd.concat([y_Crop_18,X,y_DegreeD_18,y_Rain_18],axis=1)

    ####### 2019
    y_DegreeD_18_columns_name = list(range(100,155))
    y_Rain_18_columns_name = list(range(200,255))

    y_DegreeD_19.columns = y_DegreeD_18_columns_name
    y_Rain_19.columns = y_Rain_18_columns_name

    all_2018 = pd.concat([y_Crop_19,X_2019,y_DegreeD_19,y_Rain_19],axis=1)


    # ####### 2020
    # y_DegreeD_18_columns_name = list(range(100,155))
    # y_Rain_18_columns_name = list(range(200,255))

    # y_DegreeD_20.columns = y_DegreeD_18_columns_name
    # y_Rain_20.columns = y_Rain_18_columns_name

    # all_2018 = pd.concat([y_Crop_20,X_2020,y_DegreeD_20,y_Rain_20],axis=1)


    ############# check  crop field ###############

    chosen_2018 = all_2018.loc[(all_2018['field'] == 824) & (all_2018['crop'] == 'Barley')]
    print(chosen_2018.head())
    print(chosen_2018.shape)
    ### mean ###
    mean_valuse = chosen_2018.mean(axis = 0, skipna = True)
    evi_mean = mean_valuse[2:57]
    deg_mean = mean_valuse[57:112]
    rain_mean = mean_valuse[112:167]
    print(evi_mean)
    print(deg_mean)
    print(rain_mean)
    # ### std ###
    # print(chosen_2018.std(axis = 0, skipna = True))

    ###### plot ###########
    ### option 1
    # fig, ax1 = plt.subplots()


    # color = 'tab:red'
    # ax1.set_xlabel('time (s)')
    # ax1.set_ylabel('Degree Days', color=color)
    # ax1.step(evi_mean.index, deg_mean,where="post",color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('rain (mm)', color=color)  # we already handled the x-label with ax1
    # ax2.step(evi_mean.index, rain_mean,where="post",color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()

    #### option 2

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)


    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.1))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    # p1, = host.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
    # p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
    # p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")

    p1, = host.step(evi_mean.index, deg_mean,"b-",where="post",label="Accumulated DegreeDays")
    p2, = par1.step(evi_mean.index, rain_mean,"r-",where="post",label="Accumulated Rainfall")
    p3, = par2.plot(evi_mean.index, evi_mean,"g-",label="EVI")

    # host.set_xlim(0, 2)
    host.set_ylim(0, 4000)
    par1.set_ylim(0, 800)
    par2.set_ylim(0, 1)

    host.set_xlabel("Date",fontsize=36)
    host.set_ylabel("Accumulated DegreeDays",fontsize=36)
    par1.set_ylabel("Accumulated Rainfall",fontsize=36)
    par2.set_ylabel("EVI",fontsize=36)

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y',labelsize=16, colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y',labelsize=16, colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y',labelsize=16, colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines],fontsize=20,loc="upper left")
    # plt.savefig("2018-2421-wheat.png", format="png")
    plt.show()



