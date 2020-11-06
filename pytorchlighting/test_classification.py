import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


# self attention with lstm
class AttentionModel(pl.LightningModule):
    def __init__(self, trial,batch_size,input_dim, output_dim):
        super(AttentionModel, self).__init__()
        
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden_dim = 40
        self.recurrent_layers = 2
        self.dropout_p = 0.1

        self.input_embeded = nn.Linear(input_dim, self.hidden_dim//2)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(input_size=self.hidden_dim//2, hidden_size=self.hidden_dim, num_layers=self.recurrent_layers,
                            bidirectional=True)

        self.self_attention = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim*2, 1)
        )

        self.scale = 1.0/np.sqrt(self.hidden_dim)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.output_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.label = nn.Linear(self.hidden_dim*4, output_dim)

    def forward(self, input_sentences, batch_size=None):
        input = self.dropout(torch.tanh(self.input_embeded(input_sentences)))
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
        else:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))

        self.lstm.flatten_parameters()

        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (h_0, c_0))
        output = output.permute(1, 0, 2)

        attn_ene = self.self_attention(output)
        attn_ene = attn_ene.reshape(
            self.batch_size, -1)

        # scale
        attn_ene.mul_(self.scale)
        attns = F.softmax(attn_ene, dim=1).unsqueeze(2)

        final_inputs = (output * attns).sum(dim=1)
        final_inputs2 = output.sum(dim=1)

        combined_inputs = torch.cat([final_inputs, final_inputs2], dim=1)
        logits = self.label(combined_inputs)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)



































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
    def __init__(self, trial,batch_size,input_dim, output_dim):
        super(AttentionModel, self).__init__()
        
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden_dim = trial.suggest_int("hidden_dim", 40, 80,step=4,log=False)
        self.recurrent_layers = trial.suggest_int("recurrent_layers", 1, 3)
        self.dropout_p = trial.suggest_float("dropout_rate", 0, 0.3,step=0.1,log=False)

        self.input_embeded = nn.Linear(input_dim, self.hidden_dim//2)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(input_size=self.hidden_dim//2, hidden_size=self.hidden_dim, num_layers=self.recurrent_layers,
                            bidirectional=True)

        self.self_attention = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim*2, 1)
        )

        self.scale = 1.0/np.sqrt(self.hidden_dim)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.output_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.label = nn.Linear(self.hidden_dim*4, output_dim)

    def forward(self, input_sentences, batch_size=None):

        input = self.dropout(torch.tanh(self.input_embeded(input_sentences)))
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
        else:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))

        self.lstm.flatten_parameters()

        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (h_0, c_0))
        output = output.permute(1, 0, 2)

        attn_ene = self.self_attention(output)
        attn_ene = attn_ene.reshape(
            self.batch_size, -1)

        # scale
        attn_ene.mul_(self.scale)
        attns = F.softmax(attn_ene, dim=1).unsqueeze(2)

        final_inputs = (output * attns).sum(dim=1)
        final_inputs2 = output.sum(dim=1)

        combined_inputs = torch.cat([final_inputs, final_inputs2], dim=1)
        logits = self.label(combined_inputs)

        return logits







if __name__ == "__main__":
    # sample data

    data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/cleaned_data_25753.csv'
    df_all = pd.read_csv(data_path)
    labels = df_all.iloc[:, 2]
    df_all = df_all.iloc[:, 12:210].copy()
    # remove bigger than 1
    mask = (df_all <= 1).all(axis=1)
    df_all = df_all[mask]
    labels = labels[mask]

    # print((df_all.values >1).any())
    X = df_all
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



    X_train_resampled, y_train_resampled = X_train, y_train

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

    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    weights = [1/i for i in counts_elements]
    weights[2] = weights[2]/10
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

    model = AttentionModel(BATCH_SIZE, INPUT_DIM, HID_DIM,
                           OUTPUT_DIM, RECURRENT_Layers, DROPOUT).to(device)

    print(model)
    count = count_parameters(model)
    print(count)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 40, 60])




