from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score,confusion_matrix
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

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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
    INPUT_DIM = 3
    OUTPUT_DIM = 5
    HID_DIM = 256
    DROPOUT = 0.2
    RECURRENT_Layers = 2
    LR = 0.004  # learning rate
    EPOCHS = 1
    BATCH_SIZE = 128
    NUM_CLASSES = 5
    num_gpu = 1
    datadir = "./vic_evi"  #local
    # datadir = "/afm02/Q2/Q2067/Data/DeepLearningTestData/VIC_EVI"  #hpc
    # logdir = "/afm02/Q2/Q2067/Data/DeepLearningTestData/HPC/Log/vic/1819" # hpc
    logdir = "./vic_evi/" # local

    ### start time ####
    start_time = time.time()

    ## EVI
    x_path = f'{datadir}/data/all_2020_x.csv'
    y_path = f'{datadir}/data/all_2020_y.csv'


    X = pd.read_csv(x_path,header=0)
    y = pd.read_csv(y_path,header=0)

 
    # remove below 0
    mask = (X >= 0).all(axis=1)
    X = X[mask]
    y = y[mask]

    le = LabelEncoder()
    le.fit(y)
    print(le.classes_)
    class_names = le.classes_
    y = le.transform(y)


    ## DegreeDays
    y_DegreeD_path_18 = f'{datadir}/data/all_2020_y_accumulatedD_short.csv'

    y_DegreeD_18 = pd.read_csv(y_DegreeD_path_18,header=0)

    y_DegreeD_18 = y_DegreeD_18[mask]
    ## Accumulated Rain
    y_Rain_path_18 = f'{datadir}/data/all_2020_y_accumulated_rain_final.csv'

    y_Rain_18 = pd.read_csv(y_Rain_path_18,header=0)

    y_Rain_18 = y_Rain_18[mask]


    # # split for evi, degreeday and rain, seperately
    # EVI_train, EVI_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.001, random_state=SEED, stratify=y)

    # DD_train, DD_test, _, _ = train_test_split(
    #     y_DegreeD_18, y,train_size=0.001, test_size=None, random_state=SEED, stratify=y)

    # Rain_train, Rain_test, _, _ = train_test_split(
    #     y_Rain_18, y,train_size=0.001, test_size=None, random_state=SEED, stratify=y)


    EVI_train, y_train = X, y
    DD_train = y_DegreeD_18
    Rain_train = y_Rain_18

    print('EVI_train:{}'.format(EVI_train.shape))
    # print('EVI_test:{}'.format(EVI_test.shape))
    print('DD_train:{}'.format(DD_train.shape))
    # print('DD_test:{}'.format(DD_test.shape))
    print('Rain_train:{}'.format(Rain_train.shape))
    # print('Rain_test:{}'.format(Rain_test.shape))
    print('y_train:{}'.format(y_train.shape))
    # print('y_test:{}'.format(y_test.shape))

    ############ normalization #################

    scaler_EVI = MinMaxScaler()
    scaler_EVI.fit(EVI_train)
    EVI_train = scaler_EVI.transform(EVI_train)
    # EVI_test = scaler_EVI.transform(EVI_test)

    scaler_DD = MinMaxScaler()
    scaler_DD.fit(DD_train)
    DD_train = scaler_DD.transform(DD_train)
    # DD_test = scaler_DD.transform(DD_test)
    
    scaler_Rain = MinMaxScaler()
    scaler_Rain.fit(Rain_train)
    Rain_train = scaler_Rain.transform(Rain_train)
    # Rain_test = scaler_Rain.transform(Rain_test)  


    # print('EVI_train:{}'.format(EVI_train.shape))
    # print('EVI_test:{}'.format(EVI_test.shape))
    # print('DD_train:{}'.format(DD_train.shape))
    # print('DD_test:{}'.format(DD_test.shape))
    # print('Rain_train:{}'.format(Rain_train.shape))
    # print('Rain_test:{}'.format(Rain_test.shape))
    # print('y_train:{}'.format(y_train.shape))
    # print('y_test:{}'.format(y_test.shape))

    # # # balance data
    # # ros = RandomOverSampler(random_state=SEED)
    # # X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    # X_train_resampled, y_train_resampled = X_train, y_train


    # # check sample no.
    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))


    # # prepare PyTorch Datasets
    #### EVI ########
    EVI_train_tensor = numpy_to_tensor(
        EVI_train, torch.FloatTensor)
    y_train_tensor = numpy_to_tensor(y_train, torch.long)
    # EVI_test_tensor = numpy_to_tensor(EVI_test, torch.FloatTensor)
    # y_test_tensor = numpy_to_tensor(y_test, torch.long)

    #### DD ########
    DD_train_tensor = numpy_to_tensor(
        DD_train, torch.FloatTensor)
    # DD_test_tensor = numpy_to_tensor(DD_test, torch.FloatTensor)


    #### Rain ########
    Rain_train_tensor = numpy_to_tensor(
        Rain_train, torch.FloatTensor)
    # Rain_test_tensor = numpy_to_tensor(Rain_test, torch.FloatTensor)


    EVI_train_tensor = torch.unsqueeze(EVI_train_tensor, 2)
    # EVI_test_tensor = torch.unsqueeze(EVI_test_tensor, 2)
    DD_train_tensor = torch.unsqueeze(DD_train_tensor, 2)
    # DD_test_tensor = torch.unsqueeze(DD_test_tensor, 2)
    Rain_train_tensor = torch.unsqueeze(Rain_train_tensor, 2)
    # Rain_test_tensor = torch.unsqueeze(Rain_test_tensor, 2)

    All_X_train = torch.cat((EVI_train_tensor, DD_train_tensor, Rain_train_tensor), 2)


    # All_X_train = EVI_train_tensor
    # All_X_test = EVI_test_tensor

    # All_X_train = torch.cat((EVI_train_tensor, DD_train_tensor), 2)
    # All_X_test = torch.cat((EVI_test_tensor, DD_test_tensor), 2)

    # print('All_X_train:{}'.format(All_X_train.shape))
    # print('All_X_test:{}'.format(All_X_test.shape))


    ########### Early Classification ###############

    ### 456, 18####
    ### 456789  37#####

    # All_X_train = All_X_train[:,0:37,:]
    # All_X_test = All_X_test[:,0:37,:]

    print('All_X_train:{}'.format(All_X_train.shape))
    # print('All_X_test:{}'.format(All_X_test.shape))

    train_ds = TensorDataset(All_X_train, y_train_tensor)
    # valid_ds = TensorDataset(All_X_test, y_test_tensor)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,shuffle=False, drop_last=True, num_workers=0)



    # Catalyst loader:

    loaders = OrderedDict()
    # loaders["train"] = train_dl


    # model, criterion, optimizer, scheduler

    model = AttentionModel(INPUT_DIM, HID_DIM,
                           OUTPUT_DIM, RECURRENT_Layers, DROPOUT).to(device)

    # print(model)
    # count = count_parameters(model)
    # print(count)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60])


    # # model training
    runner = SupervisedRunner()
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
    #     callbacks=[AccuracyCallback(num_classes=NUM_CLASSES, topk_args=[
    #         1, 2]), EarlyStoppingCallback(metric='accuracy01', minimize=False, patience=10)]
    # )

    # # model inference
    test_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=True, num_workers=0)

    test_truth = []
    for i in test_dl:
        test_truth.append(i[1].cpu().numpy().tolist())

    test_truth = [item for sublist in test_truth for item in sublist]

    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(model=model,
                              loader=test_dl, resume=f"{logdir}/model/3_all.pth")
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

    y_true = true_labels
    y_pred = pred_labels
    target_names = ['Barley', 'Canola', 'Chickpea', 'Lentils', 'Wheat']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=2))
    print(confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)))
    print('Kappa', cohen_kappa_score(y_true, y_pred))


    #### end time #####
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Execution Time: {epoch_mins}m {epoch_secs}s')

    # ### save predictions as csv
    # results.to_csv(f"{logdir}/predictions/predictions.csv", index=False)

    # ### generate test xy for steamlit
    # X_test = scaler.inverse_transform(X_test)
    # df_test = pd.DataFrame(X_test)

    # df_test.iloc[0:5000, :].to_csv(f"{logdir}/data/test_2019_198.csv", index=False)

    # df_test_y = pd.DataFrame(y_test)

    # df_test_y.iloc[0:5000, :].to_csv(
    #     f"{logdir}/data/test_y_2019_198.csv", index=False)
