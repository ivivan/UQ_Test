from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score
from catalyst.utils import set_global_seed
from catalyst.dl import SupervisedRunner, Runner
from catalyst.utils import metrics
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback, CriterionCallback, OptunaCallback
from catalyst.utils.checkpoint import load_checkpoint
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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# determine the supported device


def get_device():
    device = torch.device('cpu')  # don't have GPU
    return device


# def get_device():
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu')  # don't have GPU
#     return device


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
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, recurrent_layers, dropout_p):
        super(AttentionModel, self).__init__()

        self.batch_size = batch_size
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

        attn_ene = attn_ene.view(
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


    # model hyperparameters
    INPUT_DIM = 1
    OUTPUT_DIM = 7
    HID_DIM = 128
    DROPOUT = 0.2
    RECURRENT_Layers = 1
    EPOCHS = 400
    BATCH_SIZE = 1
    num_classes = 7
    lr = 0.004
    input_size = 40
    num_gpu = 1
    datadir = "R:/CROPPHEN-Q2067"  #local
    # datadir = "/afm02/Q2/Q2067"  #hpc
    logdir = "/clusterdata/uqyzha77/Log/vic/big/full"


    # generate dataloader
    # data_path_x_train = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_small/train_x_small.csv'
    # data_path_y_train = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_small/train_y_small.csv'
    # data_path_x_test = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_small/test_x_small.csv'
    # data_path_y_test = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_small/test_y_small.csv'


    data_path_x_train = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_big/train_x_big.csv'
    data_path_y_train = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_big/train_y_big.csv'
    data_path_x_test = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_big/test_x_big.csv'
    data_path_y_test = f'{datadir}/Data/DeepLearningTestData/New_NDVI_test/Test_big/test_y_big.csv'

    df_x_train = pd.read_csv(data_path_x_train)
    df_y_train = pd.read_csv(data_path_y_train, dtype=np.int32)
    df_x_test = pd.read_csv(data_path_x_test)
    df_y_test = pd.read_csv(data_path_y_test, dtype=np.int32)

    train_X, test_X, train_y, test_y = df_x_train.to_numpy(
    ), df_x_test.to_numpy(), df_y_train.to_numpy(), df_y_test.to_numpy()

    train_y = np.squeeze(train_y)
    test_y = np.squeeze(test_y)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     train_X, train_y, test_size=0.1, random_state=SEED, stratify=train_y)

    # X_train_resampled, y_train_resampled = X_train, y_train


    X_train_resampled, y_train_resampled = train_X, train_y
    X_test, y_test = test_X,test_y

    unique_elements, counts_elements = np.unique(train_y, return_counts=True)
    weights = [1/i for i in counts_elements]
    weights[2] = weights[2]/15
    print(np.asarray((unique_elements, counts_elements)))
    print(weights)
    samples_weight = np.array([weights[t] for t in train_y])
    samples_weights = torch.FloatTensor(samples_weight).to(device)
    class_weights = torch.FloatTensor(weights).to(device)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weights, len(X_train_resampled), replacement=True)

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

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, drop_last=True, num_workers=0)
    # valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
    #                       shuffle=False, drop_last=True, num_workers=0)
    valid_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=True, num_workers=0)

    # Catalyst loader:
    loaders = OrderedDict()
    loaders["train"] = train_dl
    loaders["valid"] = valid_dl

    # model
    model = AttentionModel(BATCH_SIZE//num_gpu, INPUT_DIM, HID_DIM,
                        OUTPUT_DIM, RECURRENT_Layers, DROPOUT).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 60])
    criterion = torch.nn.CrossEntropyLoss()

    # # model training
    runner = SupervisedRunner()
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=EPOCHS,
    #     verbose=True,
    #     callbacks=[
    #         AccuracyCallback(num_classes=5, topk_args=[
    #             1, 2]), EarlyStoppingCallback(metric='accuracy01', minimize=False, patience=10)
    #     ],
    # )


    # model inference

    # load_device = torch.device('cpu')

    # state = torch.load('hpc/model/best_full_big.pth',map_location=load_device)
    # model.load_state_dict(state['model_state_dict'])
    # model.to(load_device)

    # # Export the trained model to ONNX

    # dummy_input = torch.randn(1, input_size, INPUT_DIM,device=load_device) # one black and white 28 x 28 picture will be the input to the model
    # torch.onnx.export(model, dummy_input, "hpc/model/onnx_big.onnx", input_names=['vi_input'], output_names=['vi_output'])

    # dummy_output = model(dummy_input)
    # print(dummy_output)


    # model_onnx = onnx.load('hpc/model/onnx_small.onnx')

    # tf_rep = prepare(model_onnx)

    # # Export model as .pb file
    # tf_rep.export_graph('./models/model_simple.pb')

    # # Export to ONNX format
    # torch.onnx.export(model_pytorch, dummy_input, './models/model_simple.onnx', input_names=['test_input'], output_names=['test_output'])



    test_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=True, num_workers=0)

    class_names = ['0', '1', '2', '3', '4','5','6']

    test_truth = []
    for i in test_dl:
        test_truth.append(i[1].cpu().numpy().tolist())

    test_truth = [item for sublist in test_truth for item in sublist]


    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(model=model,
                              loader=test_dl, resume=f"hpc/model/best_full_big.pth")
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
    # results.columns = ['Prob_Barley', 'Prob_Canola', 'Prob_Chickpea', 'Prob_Lentils',
    #                    'Prob_Wheat', 'Pred_label', 'True_label', 'Pred_class', 'True_class']

    # classification report

    y_true = pred_labels
    y_pred = true_labels
    target_names = ['0', '1', '2', '3', '4','5','6']
    print(classification_report(y_true, y_pred, target_names=target_names,zero_division=0))
    print(f1_score(y_true, y_pred, labels=np.unique(y_pred),average='macro'))
    print('Kappa',cohen_kappa_score