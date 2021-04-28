from sklearn.metrics import classification_report, f1_score, cohen_kappa_score, accuracy_score, mean_squared_error, r2_score
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
sns.color_palette("viridis", as_cmap=True)
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

    datadir = "R:/CROPPHEN-Q2067"  # local
    logdir = "./vic120"  # local

    folder_path = f'{logdir}/data/check2.csv'
    data_120 = pd.read_csv(folder_path)

    data_120['Crop'] = data_120['Crop'].str.split(' ').str[0]

    ############# pickup crop ###############
    chosen_crop = data_120

    for z in range(7, 30, 7):

        previous_days = z
        previous_days_ac = 1
        picked__rain_t = []
        picked_rain_ac = []
        doy_sown = []
        crop_type = []

        for i in range(0, chosen_crop.shape[0]):
            # for i in range(0,1):
            crop_temp = chosen_crop.iloc[i,0]
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
            crop_type.append(crop_temp)

        d = {'picked__rain_t': picked__rain_t,
             'DOY_Sown': doy_sown, 'picked_rain_ac': picked_rain_ac, 'crop_type':crop_type}
        df = pd.DataFrame(data=d)

        doy_sown = [x * 2 for x in doy_sown]

        print(picked__rain_t)
        print(doy_sown)
        print(picked_rain_ac)
        print(crop_type)

        

        b = pd.get_dummies(crop_type)
        crop_type = b.values.argmax(1)


        # Plot outputs
        fig, ax = plt.subplots(figsize=(20, 10))

        sns.scatterplot(data=df, x="picked_rain_ac", y="picked__rain_t", hue="DOY_Sown", style="crop_type", palette='magma',ax=ax,s=100)

   
        # results = ax.scatter(picked_rain_ac, picked__rain_t, c=crop_type, s=doy_sown, alpha=0.3,
        #     cmap='viridis')
        # plt.colorbar(results)  # show color scale

        ax.set_xlabel(
            f'Accumulated Rainfall ({previous_days_ac} days before sowing date)', fontsize=28)
        ax.set_ylabel(
            f'Total Rainfall ({previous_days} days before sowing date)', fontsize=28)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=24)
        fig.tight_layout()
        plt.show()
