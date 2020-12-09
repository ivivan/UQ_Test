from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
# import pickle
np.set_printoptions(threshold=np.inf)
pd.options.display.width = 0

# reproduce
SEED = 15
set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # if batch_size is None:
        #     h_0 = Variable(torch.zeros(2*self.recurrent_layers,
        #                                self.batch_size, self.hidden_dim).to(device))
        #     c_0 = Variable(torch.zeros(2*self.recurrent_layers,
        #                                self.batch_size, self.hidden_dim).to(device))
        # else:
        #     h_0 = Variable(torch.zeros(2*self.recurrent_layers,
        #                                batch_size, self.hidden_dim).to(device))
        #     c_0 = Variable(torch.zeros(2*self.recurrent_layers,
        #                                batch_size, self.hidden_dim).to(device))

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

        print(output.shape)
        print(attns.shape)

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
    num_classes = 5
    num_gpu = 2
    # datadir = "R:/CROPPHEN-Q2067"  #local
    datadir = "/afm02/Q2/Q2067"  #hpc
    # logdir = "/clusterdata/uqyzha77/Log/vic/winter/183/"
    logdir = "/afm02/Q2/Q2067/Data/DeepLearningTestData/HPC/Log/vic/winter/198/"


    # Filtered data
    data_path = f'{datadir}/MoDS/Dabang_Sheng/Data/VIC_ready2use150000_yz_filtered80210.csv'

    df_all = pd.read_csv(data_path)
    labels = df_all.iloc[:,4].copy()
    columns_name = list(range(0,276))
    df2 = pd.DataFrame(df_all['VI_values'].str.slice(1,-1).str.split().values.tolist(),columns=columns_name,dtype=float)
    X = df2.iloc[:,0:198]
    y = labels

    le = LabelEncoder()
    le.fit(y)
    print(le.classes_)
    class_names = le.classes_
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    # # normalizeation
    # scaler = StandardScaler()
    # scaler.fit(X_train)

    # scaler_data_ = np.array([scaler.scale_, scaler.mean_, scaler.var_])
    # np.save("standard_scaler_2019filtered.npy", scaler_data_)

    # Using the standard deviation, mean and variance results from above.

    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # # balance data
    # ros = RandomOverSampler(random_state=SEED)
    # X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    X_train_resampled, y_train_resampled = X_train, y_train



    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    weights = [1/i for i in counts_elements]
    weights[2] = weights[2]/15
    print(np.asarray((unique_elements, counts_elements)))
    print(weights)
    samples_weight = np.array([weights[t] for t in y_train])
    samples_weights = torch.FloatTensor(samples_weight).to(device)
    class_weights = torch.FloatTensor(weights).to(device)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weights, len(X_train_resampled), replacement=True)

    # # check sample no.
    # unique_elements, counts_elements = np.unique(y, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))

    # prepare PyTorch Datasets

    X_train_tensor = numpy_to_tensor(
        X_train_resampled.to_numpy(), torch.FloatTensor)
    y_train_tensor = numpy_to_tensor(y_train_resampled, torch.long)
    X_test_tensor = numpy_to_tensor(X_test.to_numpy(), torch.FloatTensor)
    y_test_tensor = numpy_to_tensor(y_test, torch.long)

    X_train_tensor = torch.unsqueeze(X_train_tensor, 2)
    X_test_tensor = torch.unsqueeze(X_test_tensor, 2)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    valid_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, drop_last=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=True, num_workers=0)

    ground_truth = []
    for i in valid_dl:
        ground_truth.append(i[1].cpu().numpy().tolist())

    # print(ground_truth.flatten())
    ground_truth = [item for sublist in ground_truth for item in sublist]

    # Catalyst loader:

    loaders = OrderedDict()
    loaders["train"] = train_dl
    loaders["valid"] = valid_dl

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
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
        timeit=True,
        loaders=loaders,
        logdir=logdir,
        num_epochs=EPOCHS,
        load_best_on_end=True,
        callbacks=[AccuracyCallback(num_classes=5, topk_args=[
            1, 2]), EarlyStoppingCallback(metric='accuracy01', minimize=False, patience=10)]
    )

    # # model inference
    # test_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
    #                      shuffle=False, drop_last=True, num_workers=0)

    # test_truth = []
    # for i in test_dl:
    #     test_truth.append(i[1].cpu().numpy().tolist())

    # test_truth = [item for sublist in test_truth for item in sublist]

    # predictions = np.vstack(list(map(
    #     lambda x: x["logits"].cpu().numpy(),
    #     runner.predict_loader(model=model,
    #                           loader=test_dl, resume=f"{logdir}/checkpoints/best_full_2019_90.pth")
    # )))

    # probabilities = []
    # pred_labels = []
    # true_labels = []
    # pred_classes = []
    # true_classes = []
    # for i, (truth, logits) in enumerate(zip(test_truth, predictions)):
    #     probability = torch.softmax(torch.from_numpy(logits), dim=0)
    #     pred_label = probability.argmax().item()
    #     probabilities.append(probability.cpu().numpy())
    #     pred_labels.append(pred_label)
    #     true_labels.append(truth)
    #     pred_classes.append(class_names[pred_label])
    #     true_classes.append(class_names[truth])

    # probabilities_df = pd.DataFrame(probabilities)
    # true_labels_df = pd.DataFrame(true_labels)
    # pred_labels_df = pd.DataFrame(pred_labels)
    # pred_classes_df = pd.DataFrame(pred_classes)
    # true_classes_df = pd.DataFrame(true_classes)

    # results = pd.concat([probabilities_df, pred_labels_df, true_labels_df,
    #                      pred_classes_df, true_classes_df], axis=1)
    # results.columns = ['Prob_Barley', 'Prob_Canola', 'Prob_Chickpea', 'Prob_Lentils',
    #                    'Prob_Wheat', 'Pred_label', 'True_label', 'Pred_class', 'True_class']

    # # classification report

    # y_true = pred_labels
    # y_pred = true_labels
    # target_names = ['Barley', 'Canola', 'Chickpea', 'Lentils', 'Wheat']
    # print(classification_report(y_true, y_pred, target_names=target_names))
    # print('Kappa', cohen_kappa_score(y_true, y_pred))

    # #### save predictions as csv
    # # results.to_csv(f"{logdir}/predictions/predictions.csv", index=False)

    # generate test xy for steamlit
    # X_test = scaler.inverse_transform(X_test)
    # df_test = pd.DataFrame(X_test)

    # df_test.iloc[0:1000, :].to_csv(f"{logdir}/predictions/test_2019_filtered.csv", index=False)

    # df_test_y = pd.DataFrame(y_test)

    # df_test_y.iloc[0:1000, :].to_csv(
    #     f"{logdir}/predictions/test_y_2019_filtered.csv", index=False)
