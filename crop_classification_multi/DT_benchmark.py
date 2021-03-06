
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score,cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

    # sample data

    # data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/cleaned_data_25753.csv'
    # df_all = pd.read_csv(data_path)
    # labels = df_all.iloc[:,2]
    # df_all = df_all.iloc[:, 12:210].copy()

    data_path = 'R:/CROPPHEN-Q2067/MoDS/Dabang_Sheng/Data/VIC_ready2use150000.csv'
    df_all = pd.read_csv(data_path)
    df_all = df_all.iloc[:, 7:205].copy()
    labels = df_all.columns[1:]

    X = df_all[labels]
    y = df_all['paddocktyp']

    le = LabelEncoder()
    le.fit(y)
    print(le.classes_)
    class_names = le.classes_
    y = le.transform(y)

    # check negative values
    # print(X[(X < 0).all(1)])
    # print(X[(X > 1).all(1)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    # balance data
    ros = RandomOverSampler(random_state=SEED)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # check sample no.
    # unique_elements, counts_elements = np.unique(y_test, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))

    # prepare PyTorch Datasets

    # DT

    modelo = RandomForestClassifier(n_estimators=1)
    modelo_GB = GradientBoostingClassifier()

    modelo.fit(X_train_resampled, y_train_resampled)



    # read and prepare data
    vic2020_folder = 'R:/CROPPHEN-Q2067/Data/DeepLearningTestData/NDVI/VIC2020_120'
    alldata, allpaths = data_together(vic2020_folder)


    frames = [alldata[0], alldata[1], alldata[2]]
    vic2020 = pd.concat(frames)

    labels = vic2020.iloc[:,2]
    df_all = vic2020.iloc[:, 3:].copy()
    column_list = np.arange(198)
    df_all.columns = column_list
    X = df_all
    y = labels


    # le = LabelEncoder()
    # le.fit(y)
    # print(le.classes_)
    # class_names = le.classes_
    # y = le.transform(y)

    le = LabelEncoder()
    le.fit(['Barley', 'Canola', 'Chick Pea', 'Lentils', 'Wheat'])
    print(le.classes_)
    class_names = le.classes_
    y = le.transform(y)


    X_test = X
    y_test = y





    predections = modelo.predict(X_test)

    target_names = ['Barley', 'Canola', 'ChickPea', 'Lentils', 'Wheat']
    print(classification_report(y_test, predections, target_names=target_names))
    print('Kappa',cohen_kappa_score(y_test,predections))


    # X_train_tensor = numpy_to_tensor(
    #     X_train_resampled.to_numpy(), torch.FloatTensor)
    # y_train_tensor = numpy_to_tensor(y_train_resampled, torch.long)
    # X_test_tensor = numpy_to_tensor(X_test.to_numpy(), torch.FloatTensor)
    # y_test_tensor = numpy_to_tensor(y_test, torch.long)

    # X_train_tensor = torch.unsqueeze(X_train_tensor, 2)
    # X_test_tensor = torch.unsqueeze(X_test_tensor, 2)

    # train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    # valid_ds = TensorDataset(X_test_tensor, y_test_tensor)

    # # DataLoader definition
    # # model hyperparameters
    # INPUT_DIM = 1
    # OUTPUT_DIM = 5
    # HID_DIM = 64
    # DROPOUT = 0.1
    # RECURRENT_Layers = 2
    # LR = 0.001  # learning rate
    # EPOCHS = 300
    # BATCH_SIZE = 100
    # num_classes = 5

    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
    #                       shuffle=True, drop_last=True, num_workers=0)
    # valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
    #                       shuffle=False, drop_last=True, num_workers=0)

    # ground_truth = []
    # for i in valid_dl:
    #     ground_truth.append(i[1].cpu().numpy().tolist())

    # # print(ground_truth.flatten())
    # ground_truth = [item for sublist in ground_truth for item in sublist]

    # # Catalyst loader:

    # loaders = OrderedDict()
    # loaders["train"] = train_dl
    # loaders["valid"] = valid_dl

    # # model, criterion, optimizer, scheduler

    # model = AttentionModel(BATCH_SIZE, INPUT_DIM, HID_DIM,
    #                        OUTPUT_DIM, RECURRENT_Layers, DROPOUT).to(device)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

    # # model training
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
    #     callbacks=[EarlyStoppingCallback(patience=10), dl.AlchemyLogger(
    #         token="008cad539a5bd45e588162400272a17e",  # your Alchemy token
    #         project="cropclassification",
    #         experiment="lstm_attention",
    #         group="UQ",
    #     )]
    # )

    # # # model training
    # # runner = dl.SupervisedRunner()
    # # logdir = "./logdir"
    # # runner.train(
    # #     model=model,
    # #     criterion=criterion,
    # #     optimizer=optimizer,
    # #     scheduler=scheduler,
    # #     verbose=True,
    # #     timeit=True,
    # #     loaders=loaders,
    # #     logdir=logdir,
    # #     num_epochs=EPOCHS,
    # #     load_best_on_end=True,
    # #     callbacks=[CriterionCallback(input_key="targets", output_key="logits", prefix='loss'), AccuracyCallback(num_classes=5, topk_args=[
    # #         1, 2], input_key='targets', output_key='logits'), EarlyStoppingCallback(patience=10)]
    # # )

    # # #### model inference
    # # test_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
    # #                      shuffle=False, drop_last=True, num_workers=0)

    # # test_truth = []
    # # for i in test_dl:
    # #     test_truth.append(i[1].cpu().numpy().tolist())

    # # test_truth = [item for sublist in test_truth for item in sublist]

    # # predictions = np.vstack(list(map(
    # #     lambda x: x["logits"].cpu().numpy(),
    # #     runner.predict_loader(model=model,
    # #                           loader=test_dl, resume=f"{logdir}/checkpoints/best_full.pth")
    # # )))

    # # probabilities = []
    # # pred_labels = []
    # # true_labels = []
    # # pred_classes = []
    # # true_classes = []
    # # for i, (truth, logits) in enumerate(zip(test_truth, predictions)):
    # #     probability = torch.softmax(torch.from_numpy(logits), dim=0)
    # #     pred_label = probability.argmax().item()
    # #     probabilities.append(probability.cpu().numpy())
    # #     pred_labels.append(pred_label)
    # #     true_labels.append(truth)
    # #     pred_classes.append(class_names[pred_label])
    # #     true_classes.append(class_names[truth])

    # # probabilities_df = pd.DataFrame(probabilities)
    # # true_labels_df = pd.DataFrame(true_labels)
    # # pred_labels_df = pd.DataFrame(pred_labels)
    # # pred_classes_df = pd.DataFrame(pred_classes)
    # # true_classes_df = pd.DataFrame(true_classes)

    # # results = pd.concat([probabilities_df, pred_labels_df, true_labels_df,
    # #                      pred_classes_df, true_classes_df], axis=1)
    # # results.columns = ['Prob_Barley', 'Prob_Canola', 'Prob_Chick_Pea', 'Prob_Lentils',
    # #                    'Prob_Wheat', 'Pred_label', 'True_label', 'Pred_class', 'True_class']

    # # #### classification report

    # # y_true = pred_labels
    # # y_pred = true_labels
    # # target_names = ['Barley', 'Canola', 'Chick Pea', 'Lentils', 'Wheat']
    # # print(classification_report(y_true, y_pred, target_names=target_names))

    # # #### save predictions as csv
    # # # results.to_csv(f"{logdir}/predictions/predictions.csv", index=False)
