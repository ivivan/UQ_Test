from LSTM_Classification_V3 import AttentionModel, clip_gradient
# from LSTM_Classification import AttentionModel, clip_gradient
from loss import LabelSmoothingCrossEntropy
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score,cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score
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
import os
import time
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
pd.options.display.width = 0


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

# read saved scaler for inference
def read_scaler(path):
    scaler_data_ = np.load(path)
    return scaler_data_

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
    # labels = df_all.iloc[:,2]
    # df_all = df_all.iloc[:, 11:].copy()
    # X = df_all
    # y = labels


    # data_path2 = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/VIC_ready2use150000.csv'
    # df_all2 = pd.read_csv(data_path2)
    # df_all2 = df_all2.iloc[:, 7:].copy()
    # labels2 = df_all2.columns[1:]

    # common_cols = list(set(df_all2.columns) & set(df_all.columns))
    # print(df_all.columns)
    # print(df_all2.columns)
    # print(common_cols)
    # checkpd = pd.merge(df_all2, df_all, on=list(df_all2.columns), how='inner')
    # print(checkpd)
    # print(checkpd.shape[0])

    # # # pick up only NDVI,and paddocktyp
    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/VIC_ready2use150000.csv'
    # df_all = pd.read_csv(data_path)
    # df_all = df_all.iloc[:, 6:].copy()
    # labels = df_all.columns[1:]

    # X = df_all[labels]
    # y = df_all['paddocktyp']



    # VIC 2020 test data

    # read and prepare data
    vic2020_folder = 'R:/CROPPHEN-Q2067/Data/DeepLearningTestData/NDVI/VIC2020_120'
    alldata, allpaths = data_together(vic2020_folder)


    frames = [alldata[0], alldata[1], alldata[2]]
    vic2020 = pd.concat(frames)

    labels = vic2020.iloc[:,2]
    df_all = vic2020.iloc[:, 3:].copy()
    column_list = np.arange(198)
    df_all.columns = column_list

    # remove bigger than 1
    mask = (df_all <= 1).all(axis=1)
    df_all = df_all[mask]
    labels = labels[mask]

    X = df_all
    y = labels


    # le = LabelEncoder()
    # le.fit(y)
    # print(le.classes_)
    # class_names = le.classes_
    # y = le.transform(y)

    le = LabelEncoder()
    le.fit(['Barley', 'Canola', 'Chickpea', 'Lentils', 'Wheat'])
    print(le.classes_)
    class_names = le.classes_
    y = le.transform(y)




    # # check sample no.
    # unique_elements, counts_elements = np.unique(y, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.3, random_state=SEED, stratify=y)



    # # normalizeation
    # scaler = StandardScaler()
    # scaler.fit(X)
    # scaler_info = read_scaler('./standard_scaler_2019filtered.npy')
    # scaler.scale_, scaler.mean_, scaler.var_ = scaler_info[0][:198], scaler_info[1][:198], scaler_info[2][:198]
    # # Using the standard deviation, mean and variance results from above.
    # X_test = scaler.transform(X)
    
    X_test = X

    # prepare PyTorch Datasets
    X_test_tensor = numpy_to_tensor(X_test.to_numpy(), torch.FloatTensor)
    y_test_tensor = numpy_to_tensor(y, torch.long)
    X_test_tensor = torch.unsqueeze(X_test_tensor, 2)
    valid_ds = TensorDataset(X_test_tensor,y_test_tensor)

    
    # DataLoader definition
    # model hyperparameters
    INPUT_DIM = 1
    OUTPUT_DIM = 5
    HID_DIM = 64
    DROPOUT = 0.1
    RECURRENT_Layers = 2
    # LR = 0.001  # learning rate
    EPOCHS = 400
    BATCH_SIZE = 32
    num_classes = 5

    # model, criterion, optimizer, scheduler

    model = AttentionModel(BATCH_SIZE, INPUT_DIM, HID_DIM,
                           OUTPUT_DIM, RECURRENT_Layers, DROPOUT).to(device)

    print(model)
    count = count_parameters(model)
    print(count)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 40, 60])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 60, 90])

    # model training
    # runner = CustomRunner()
    # logdir = "./logdir"
    # runner.train(
    #     model=model,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     num_epochs=EPOCHS,
    #     loaders=loaders,
    #     logdir=logdir,
    #     verbose=True,
    #     timeit=True,
    #     callbacks=[EarlyStoppingCallback(patience=10)]
    # )

    # # model training
    runner = SupervisedRunner()
    logdir = "./crop_classification_multi/checkpoints"
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     verbose=True,
    #     timeit=True,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=EPOCHS,
    #     load_best_on_end=True,
    #     callbacks=[AccuracyCallback(num_classes=5, topk_args=[
    #         1, 2]), EarlyStoppingCallback(patience=10)]
    # )

    # model inference
    test_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=True, num_workers=0)

    test_truth = []
    for i in test_dl:
        test_truth.append(i[1].cpu().numpy().tolist())

    test_truth = [item for sublist in test_truth for item in sublist]


    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(model=model,
                              loader=test_dl, resume=f"{logdir}/best_full_2019_210.pth")
    )))

    probabilities = []
    pred_labels = []
    true_labels = []
    pred_classes = []
    true_classes = []
    for i, (truth, logits) in enumerate(zip(test_truth, predictions)):
        probability = torch.softmax(torch.from_numpy(logits), dim=0)
        pred_label = probability.argmax().item()
        probabilities.append(probability.cpu().numpy())
        pred_labels.append(pred_label)
        true_labels.append(truth)
        pred_classes.append(class_names[pred_label])
        true_classes.append(class_names[truth])

    probabilities_df = pd.DataFrame(probabilities)
    true_labels_df = pd.DataFrame(true_labels)
    pred_labels_df = pd.DataFrame(pred_labels)
    pred_classes_df = pd.DataFrame(pred_classes)
    true_classes_df = pd.DataFrame(true_classes)

    results = pd.concat([probabilities_df, pred_labels_df, true_labels_df,
                         pred_classes_df, true_classes_df], axis=1)
    results.columns = ['Prob_Barley', 'Prob_Canola', 'Prob_Chickpea', 'Prob_Lentils',
                       'Prob_Wheat', 'Pred_label', 'True_label', 'Pred_class', 'True_class']

    # classification report

    y_true = pred_labels
    y_pred = true_labels
    target_names = ['Barley', 'Canola', 'Chickpea', 'Lentils', 'Wheat']
    print(classification_report(y_true, y_pred, target_names=target_names,zero_division=0))
    print(f1_score(y_true, y_pred, labels=np.unique(y_pred),average='macro'))
    print('Kappa',cohen_kappa_score(y_true,y_pred))

    

    # #### save predictions as csv
    # # results.to_csv(f"{logdir}/predictions/predictions.csv", index=False)

    # # generate test xy for steamlit
    # X_test = scaler.inverse_transform(X_test)
    # df_test = pd.DataFrame(X_test)

    # df_test.iloc[0:1000, :].to_csv(f"{logdir}/predictions/test_new2020.csv", index=False)

    # df_test_y = pd.DataFrame(y)

    # df_test_y.iloc[0:1000, :].to_csv(
    #     f"{logdir}/predictions/test_y_new2020.csv", index=False)
