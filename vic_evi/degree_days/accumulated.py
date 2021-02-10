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
    datadir = "R:/CROPPHEN-Q2067"  # local
    # datadir = "/afm02/Q2/Q2067"  #hpc
    # logdir = "/clusterdata/uqyzha77/Log/vic/winter/" # hpc
    logdir = "./vic_evi/"  # local

    ##### read degreeday data ################

    degreeD_path = f'{logdir}/data/all_2020_y_degreeD_short.csv'
    dd_df = pd.read_csv(degreeD_path)

    ############ accumulate data #################
    # start Apr 1, to Dec 31        
    columns_name = list(range(0, 275))
    df2 = pd.DataFrame(dd_df['degreeDay'].str.slice(
        1, -1).str.split(',').values.tolist(), columns=columns_name, dtype=float)

    dd_np = df2.to_numpy()

    dd_resample = []

    for index, v in zip(range(0, dd_np.shape[0]), dd_np):
        dd_np[index] = np.add.accumulate(v)

    for index, v in zip(range(0, dd_np.shape[0]), dd_np):
        dd_resample.append(v[::5])

    final = np.asarray(dd_resample)
    final_df = pd.DataFrame.from_records(final)
    


    final_df.to_csv(
        f"{logdir}/data/all_2020_y_accumulatedD_short.csv", index=False)
