from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score,mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import ast
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



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
    logdir = "./vic120" # local


    folder_path = f'{logdir}/data/check.csv'
    data_120 = pd.read_csv(folder_path)

    data_120['Crop'] = data_120['Crop'].str.split(' ').str[0]





    ############# pickup crop ###############
    chosen_crop =data_120



    for z in range(1,30,5):
        for j in range(1,30,5):


            previous_days = z
            previous_days_ac = j
            picked__rain_t = []
            picked_rain_ac = []
            doy_sown = []

            for i in range(0,chosen_crop.shape[0]):
            # for i in range(0,1):
                doy_temp = chosen_crop.iloc[i, 1]

                rain_sub_temp = chosen_crop.iloc[i, 2]

                # rain_accu_temp = chosen_crop.iloc[i, 3]
                
                l = ast.literal_eval(rain_sub_temp)
                l_array = np.asarray(l)
                picked_rain_t = np.sum(l_array[-previous_days:])

                rain_sub_accumulated_temp = chosen_crop.iloc[i, 3]
                l2 = ast.literal_eval(rain_sub_accumulated_temp)
                l2_array = np.asarray(l2)


                picked_rain_ac.append(l2_array[-previous_days_ac])
                picked__rain_t.append(picked_rain_t)
                doy_sown.append(doy_temp)

            
            print(picked__rain_t)
            print(doy_sown)
            print(picked_rain_ac)



            x = np.arange(len(picked__rain_t))  # the label locations
            width = 0.35  # the width of the bars
            average = np.average(picked__rain_t)

            # Plot outputs
            fig, ax = plt.subplots(figsize=(20, 10))
            rects1 = ax.scatter(picked_rain_ac,picked__rain_t,c='r')
            # ax.axhline(y=average, color='r', linestyle='-',label=average)



            ax.set_xlabel(f'Accumulated Rainfall ({previous_days_ac} days before sowing date)',fontsize=28)
            ax.set_ylabel(f'Total Rainfall ({previous_days} days before sowing date)',fontsize=28)
            ax.legend(fontsize=24)
            

            plt.show()
    