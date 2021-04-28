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

    logdir = "./vic_evi/"  # local

    
    logits_pred_path = f'{logdir}/predictions/predictions.csv'

    logits_pred = pd.read_csv(logits_pred_path,header=0)


    ########## pick up correct predictions ###########

    ##### ['Barley' 'Canola' 'Chickpea' 'Lentils' 'Wheat'] #######
    correct_logits = logits_pred[logits_pred['Pred_label'] == logits_pred['True_label']]


    ###### check each crop ########
    select_wheat = correct_logits[correct_logits['True_label'] == 4].iloc[:30,0:5]
    select_wheat.reset_index(inplace=True)

    select_wheat = select_wheat.iloc[:,1:]


    # colnames = list (select_wheat.columns)
    # select_wheat.reset_index().plot(x="index", y=colnames[1:], kind = 'line', legend=False, 
    #                 subplots = True, sharex = True, figsize = (5.5,4), ls="none", marker="o")

    # plt.show()



    # select_wheat.plot.bar(stacked=True)
    # plt.tight_layout()
    # plt.show()




    # ################ when make the correct prediciton ###################
    # sns.distplot(select_wheat['Prob_Wheat'],  kde=False, label='Wheat')
    # sns.distplot(select_wheat['Prob_Barley'],  kde=False,label='Barley')

    # # sns.histplot(data=select_wheat, x="Prob_Wheat", color="skyblue", label="Wheat", kde=True)
    # # sns.histplot(data=select_wheat, x="Prob_Lentils", color="red", label="Lentils", kde=False)


    # plt.legend()
    # plt.ylabel('count')
    # plt.xlabel('probability')
    # plt.show()


    ################ when make the incorrect prediciton ###################
    select_wheat['Max'] = select_wheat.idxmax(axis=1)

    ############### check each type ##########

    incorrect_logits = select_wheat[select_wheat['Max'] == 'Prob_Barley']
    incorrect_logits.reset_index(inplace=True)
    incorrect_logits.drop(['index'], axis=1,inplace=True)

    incorrect_logits.plot(subplots=True)
    plt.tight_layout()
    plt.show()



 




