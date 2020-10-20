# import transformer
# from data_load import load_train_data, next_batch
from misc import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Embeddings
import argparse
from transformer import TransformerEncoder
from transformer_encoder import *
import math
import copy


from LSTM_Classification_V3 import AttentionModel, clip_gradient
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
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
# torch.cuda.empty_cache()

def make_model(src_dim, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1, batch_size=10, n_class=15):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = TransformerEncoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_dim), c(position)),
        batch_size,
        d_model,
        n_class
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


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


class CustomRunner(Runner):

    def _handle_batch(self, batch):
        x, y = batch

        # y_hat, attention = self.model(x)
        outputs = self.model(x, None)

        print('outputs',outputs.shape)
        print('outputs_v',outputs)

        loss = F.cross_entropy(outputs, y)

        print('loss',loss)


        accuracy01, accuracy02 = metrics.accuracy(
            outputs, y, topk=(1, 2))
        self.batch_metrics = {
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy02": accuracy02,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


if __name__ == '__main__':
    # sample data

    data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/VIC_ready2use150000.csv'
    df_all = pd.read_csv(data_path)

    # pick up only NDVI,and paddocktyp

    df_all = df_all.iloc[:, 6:].copy()
    labels = df_all.columns[1:]

    X = df_all[labels]

    y = df_all['paddocktyp']

    # unique_elements, counts_elements = np.unique(y, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))

    le = LabelEncoder()
    le.fit(y)
    print(le.classes_)
    class_names = le.classes_
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    # normalizeation
    scaler = StandardScaler()
    scaler.fit(X_train)

    # np.save('./scaler.scale_.npy', scaler.scale_)
    # np.save('./scaler.mean_.npy', scaler.mean_)
    # np.save('./scaler.var_.npy', scaler.var_)
    # Using the standard deviation, mean and variance results from above.

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # balance data
    ros = RandomOverSampler(random_state=SEED)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    # X_train_resampled, y_train_resampled = X_train, y_train

    # # check sample no.
    # unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))

    # weights = [np.amax(counts_elements)/i for i in counts_elements]
    # class_weights = torch.FloatTensor(weights).to(device)

    # prepare PyTorch Datasets

    X_train_tensor = numpy_to_tensor(
        X_train_resampled, torch.FloatTensor)
    y_train_tensor = numpy_to_tensor(y_train_resampled, torch.long)
    X_test_tensor = numpy_to_tensor(X_test, torch.FloatTensor)
    y_test_tensor = numpy_to_tensor(y_test, torch.long)

    X_train_tensor = torch.unsqueeze(X_train_tensor, 2)
    X_test_tensor = torch.unsqueeze(X_test_tensor, 2)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    valid_ds = TensorDataset(X_test_tensor, y_test_tensor)

    # # DataLoader definition
    # # model hyperparameters
    INPUT_DIM = 1
    OUTPUT_DIM = 5
    MODEL_DIM = 64
    HID_DIM = 40
    DROPOUT = 0.1
    LR = 0.004  # learning rate
    EPOCHS = 2
    BATCH_SIZE = 32
    N_CLASSES = 5
    ATTENTION_LAYER = 6
    FF_DIM = 256
    HEAD = 2

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True, num_workers=0)
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



    model = make_model(INPUT_DIM, N=2, d_model=MODEL_DIM, d_ff=FF_DIM,h=HEAD, dropout=DROPOUT,
                       batch_size=BATCH_SIZE, n_class=N_CLASSES)
    model = model.to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 25, 40])

    # # epochs = 10
    # for iter in range(args.num_iterations):
    #     x, y = next_batch(X, args.batch_size, args.dim_model)
    # # batches = get_batches(in_text, out_text, 10, 200)
    # # for x, y in batches:
    #     x = torch.tensor(x, dtype=torch.float, device=device)
    #     y = torch.tensor(y, dtype=torch.long, device=device)
    #     y = torch.squeeze(y)
    #     criterion, optimizer = get_criterion(model)
    #     optimizer.zero_grad()
    #     output = model(x, None)
    #     loss = criterion(output,y)
    #     if iter % 100 == 0:
    #         _, preds = torch.max(output, dim=1)
    #         print("Iteration {}: loss= {}, accuracy= {}".format(iter,loss.item(), float(torch.sum(preds==y).item()/args.batch_size)))
    #     loss.backward()
    #     optimizer.step()

    # model training
    runner = CustomRunner()
    logdir = "./logdir"
    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        loaders=loaders,
        logdir=logdir,
        verbose=True,
        timeit=True,
        callbacks=[EarlyStoppingCallback(patience=10)]
    )

    # # model training
    # runner = SupervisedRunner()
    # logdir = "./logdir"
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
    #                           loader=test_dl, resume=f"{logdir}/checkpoints/best_full.pth")
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
    # results.columns = ['Prob_Barley', 'Prob_Canola', 'Prob_Chick_Pea', 'Prob_Lentils',
    #                    'Prob_Wheat', 'Pred_label', 'True_label', 'Pred_class', 'True_class']

    # # classification report

    # y_true = pred_labels
    # y_pred = true_labels
    # target_names = ['Barley', 'Canola', 'Chick Pea', 'Lentils', 'Wheat']
    # print(classification_report(y_true, y_pred, target_names=target_names))

    # #### save predictions as csv
    # # results.to_csv(f"{logdir}/predictions/predictions.csv", index=False)

    # generate test xy for steamlit
    # X_test = scaler.inverse_transform(X_test)
    # df_test = pd.DataFrame(X_test)

    # df_test.iloc[0:10, :].to_csv(f"{logdir}/predictions/test.csv", index=False)

    # df_test_y = pd.DataFrame(y_test)

    # df_test_y.iloc[0:10, :].to_csv(
    #     f"{logdir}/predictions/test_y.csv", index=False)
