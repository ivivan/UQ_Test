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



def objective(trial):
    logdir = "/clusterdata/uqyzha77/Log/vic/"
    num_epochs = 100
    INPUT_DIM = 1
    OUTPUT_DIM = 5
    BATCH_SIZE = 64    # change here for multi gpu training 16*4=64
    num_classes = 5
    num_gpu = 1

    lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)

    # generate dataloader
    data_path = '/afm02/Q2/Q2067/MoDS/Dabang_Sheng/Data/VIC_ready2use150000_yz_filtered80210.csv'
    df_all = pd.read_csv(data_path)

    labels = df_all.iloc[:,4].copy()
    columns_name = list(range(0,276))
    df2 = pd.DataFrame(df_all['VI_values'].str.slice(1,-1).str.split().values.tolist(),columns=columns_name,dtype=float)
    X = df2
    y = labels

    le = LabelEncoder()
    le.fit(y)
    print(le.classes_)
    class_names = le.classes_
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    X_train_resampled, y_train_resampled = X_train, y_train



    


    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    weights = [1/i for i in counts_elements]
    weights[2] = weights[2]/15
    print(np.asarray((unique_elements, counts_elements)))
    print(weights)
    samples_weight = np.array([weights[t] for t in y_train])
    samples_weights = torch.FloatTensor(samples_weight).to(device)
    class_weights = torch.FloatTensor(weights).to(device)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(X_train_resampled),replacement=True)

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

    # Catalyst loader:
    loaders = OrderedDict()
    loaders["train"] = train_dl
    loaders["valid"] = valid_dl

 
    # model
    model = AttentionModel(trial, BATCH_SIZE//num_gpu, INPUT_DIM,
                            OUTPUT_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20,40,60])
    criterion = torch.nn.CrossEntropyLoss()

    # model training
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        callbacks=[
            AccuracyCallback(num_classes=num_classes),
            CatalystPruningCallback(
                trial, metric="accuracy01"
            ),  # top-1 accuracy as metric for pruning
        ],
    )

    return runner.state.valid_metrics["accuracy01"]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Catalyst example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Value: {}".format(trial.value))

    print("Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))