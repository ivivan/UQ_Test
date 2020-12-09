import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score

import numpy as np
import os
import io
import time
import pandas as pd
tf.config.run_functions_eagerly(True)

# class AttentionModel(tf.keras.Model):
#     def __init__(self, batch_size, input_dim, hidden_dim, output_dim, recurrent_layers, dropout_p):
#         super(AttentionModel, self).__init__()

#         self.batch_size = batch_size
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.input_dim = input_dim
#         self.recurrent_layers = recurrent_layers
#         self.dropout_p = dropout_p

#         self.input_embeded = tf.keras.layers.Dense(
#             hidden_dim//2, activation='tanh')

#         self.dropout = tf.keras.layers.Dropout(dropout_p)

#         self.rnn_layers = []
#         for _ in range(0, recurrent_layers):

#             rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim,
#                                                                            return_sequences=True,
#                                                                            return_state=True))
#             self.rnn_layers.append(rnn_layer)

#         self.self_attention = tf.keras.Sequential([
#             tf.keras.layers.Dense(hidden_dim*2, activation='relu'),
#             tf.keras.layers.Dense(1)
#         ])

#         self.scale = 1.0/np.sqrt(hidden_dim)

#         self.output_linear = tf.keras.layers.Dense(self.hidden_dim)
#         self.label = tf.keras.layers.Dense(output_dim)

#     def call(self, input_sentences, batch_size=None):
        
#         # tf.print('input1',input_sentences)

#         input = self.input_embeded(input_sentences)
#         input = self.dropout(input)

#         # tf.print('changed input',input)

#         for i, _ in enumerate(self.rnn_layers):
#             output, forward_h, forward_c, backward_h, backward_c = self.rnn_layers[i](
#                 input)

#         attn_ene = self.self_attention(output)

#         # scale
#         attn_ene = tf.math.scalar_mul(self.scale, attn_ene)
#         attns = tf.nn.softmax(attn_ene, axis=1)

#         context_vector = attns * output
#         final_inputs = tf.reduce_sum(context_vector, axis=1)
#         final_inputs2 = tf.reduce_sum(output, axis=1)

#         combined_inputs = tf.concat([final_inputs, final_inputs2], axis=1)

#         logits = self.label(combined_inputs)

#         return logits



# model hyperparameters
INPUT_DIM = 1
OUTPUT_DIM = 7
HID_DIM = 128
DROPOUT = 0.2
RECURRENT_Layers = 1
EPOCHS = 400
BATCH_SIZE = 256
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



# # model
# model = AttentionModel(BATCH_SIZE//num_gpu, INPUT_DIM, HID_DIM,
#                     OUTPUT_DIM, RECURRENT_Layers, DROPOUT)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 60])
# criterion = torch.nn.CrossEntropyLoss()


# model inference



# Export the trained model to ONNX

# dummy_input = torch.randn(1, input_size, INPUT_DIM,device=load_device) # one black and white 28 x 28 picture will be the input to the model
# torch.onnx.export(model, dummy_input, "hpc/model/onnx_small.onnx")


model_onnx = onnx.load('hpc/model/onnx_big.onnx')

tf_rep = prepare(model_onnx)


# # Export model as .pb file
# tf_rep.export_graph('hpc/model/tf_big.pb')




# X_test = np.expand_dims(X_test.astype(np.float32), axis=(2))


# # input = np.expand_dims(X_test[5000].astype(np.float32), axis=(0,2))
# # print(input.shape)

# # output = tf_rep.run(input)
# # print('The digit is classified as ', np.argmax(output))

# # no = 0
# # predictions = []
# # for i in X_test:
# #     input = np.expand_dims(i.astype(np.float32), axis=(0,2))
# #     output = np.argmax(tf_rep.run(input))
# #     predictions.append(output)
# #     print(no)
# #     no += 1



# iter_per_epoch = int(np.ceil(X_test.shape[0] * 1. / BATCH_SIZE))
# # perm_idx = np.random.permutation(X_test.shape[0])
# perm_idx = np.arange(X_test.shape[0])

# predictions = []

# for t_i in range(0, X_test.shape[0], BATCH_SIZE):
#     batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

#     x_test_batch = np.take(X_test, batch_idx, axis=0)
#     y_test_batch = np.take(y_test, batch_idx, axis=0)

#     temp = tf_rep.run(x_test_batch)
#     output = np.argmax(temp[0],axis=1)
#     predictions.append(output.tolist())
    
# predictions = [item for sublist in predictions for item in sublist]





# class_names = ['0', '1', '2', '3', '4','5','6']
# pred_labels = []
# true_labels = []
# pred_classes = []
# true_classes = []



# for i, (truth, preds) in enumerate(zip(y_test, predictions)):
#     pred_labels.append(preds)
#     true_labels.append(truth)
#     pred_classes.append(class_names[preds])
#     true_classes.append(class_names[truth])



# # classification report

# y_true = pred_labels
# y_pred = true_labels
# target_names = ['0', '1', '2', '3', '4','5','6']
# print(classification_report(y_true, y_pred, target_names=target_names,zero_division=0))
# print(f1_score(y_true, y_pred, labels=np.unique(y_pred),average='macro'))
# print('Kappa',cohen_kappa_score(y_true,y_pred))








































