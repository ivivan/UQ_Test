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

    datadir = "R:/CROPPHEN-Q2067"  #local
    logdir = "./vic_evi/" # local



    # sample data

    # ## EVI
    # x_path = './vic_evi/data/all_2018_x.csv'
    # y_path = './vic_evi/data/all_2018_y.csv'
    # x_2019_path = './vic_evi/data/all_2019_x.csv'
    # y_2019_path = './vic_evi/data/all_2019_y.csv'

    # X = pd.read_csv(x_path,header=0)
    # y = pd.read_csv(y_path,header=0)
    # X_2019 = pd.read_csv(x_2019_path,header=0)
    # y_2019 = pd.read_csv(y_2019_path,header=0)
    # X_1819 = pd.concat([X,X_2019])
    # y_1819 = pd.concat([y,y_2019])


    # ######## EVI ###########
    # X.iloc[0:6000,:].to_csv(
    #     f"{logdir}/data/visulization/all_2018_x_sample.csv", index=False)

    # X_2019.iloc[0:6000,:].to_csv(
    #     f"{logdir}/data/visulization/all_2019_x_sample.csv", index=False)




    # ## DegreeDays
    # y_DegreeD_path_18 = './vic_evi/data/all_2018_y_accumulatedD.csv'
    # y_DegreeD_path_19 = './vic_evi/data/all_2019_y_accumulatedD.csv'


    # y_DegreeD_18 = pd.read_csv(y_DegreeD_path_18,header=0)
    # y_DegreeD_19 = pd.read_csv(y_DegreeD_path_19,header=0)
    # y_DegreeD_1819 = pd.concat([y_DegreeD_18,y_DegreeD_19])


    # y_18_sample = y_DegreeD_18.T

    # plt.step(y_18_sample.index, y_18_sample.iloc[:,3000],where="post")
    # plt.show()



    ## Accumulated Rain
    y_Rain_path_18 = './vic_evi/data/all_2018_y_accumulated_rain_final.csv'
    y_Rain_path_19 = './vic_evi/data/all_2019_y_accumulated_rain_final.csv'


    y_Rain_18 = pd.read_csv(y_Rain_path_18,header=0)
    y_Rain_19 = pd.read_csv(y_Rain_path_19,header=0)
    y_Rain_1819 = pd.concat([y_Rain_18,y_Rain_19])


    y_18_sample = y_Rain_18.T

    plt.step(y_18_sample.index, y_18_sample.iloc[:,3000],where="post")
    plt.show()