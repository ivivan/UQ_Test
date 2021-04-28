from sklearn.metrics import classification_report,f1_score,cohen_kappa_score,accuracy_score,mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import ast
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



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

    datadir = "R:/CROPPHEN-Q2067"  #local
    logdir = "./vic120" # local


    folder_path = f'{logdir}/data/check.csv'
    data_120 = pd.read_csv(folder_path)

    data_120['Crop'] = data_120['Crop'].str.split(' ').str[0]


    ############# pickup crop ###############
    chosen_crop =data_120.loc[(data_120['Crop']=='Oats')]

    x_t = []
    x_a = []
    y_sown = []

    for i in range(0,chosen_crop.shape[0]):
        doy_temp = chosen_crop.iloc[i, 1]
        y_sown.append(doy_temp)

        rain_sub_temp = chosen_crop.iloc[i, 2]
        l = ast.literal_eval(rain_sub_temp)
        resampled_5days = [sum(l[x:x+5]) for x in range(0, len(l), 5)]
        l_array = np.asarray(resampled_5days)
        l_array_apr = l_array[19:]

        rain_sub_accumulated_temp = chosen_crop.iloc[i, 3]
        l2 = ast.literal_eval(rain_sub_accumulated_temp)
        resampled_2 = l2[::5]
        l2_array = np.asarray(resampled_2)
        l2_array_apr = l2_array[19:]

        x_t_temp = []
        a_t_temp = []

        for t,a in zip(l_array_apr,l2_array_apr):
            x_t_temp.append(t)
            a_t_temp.append(a)

        x_t.append(x_t_temp)
        x_a.append(a_t_temp)


    inputs = []
    for i in range (0,len(y_sown)):
        single_field = []
        for j in range (0,len(x_t[i])):
            temp = []
            temp.append(x_t[i][j])
            temp.append(x_a[i][j])
            single_field.append(temp)
        inputs.append(single_field)


    # print(len(inputs))
    

    

    y_sown = np.asarray(y_sown)
    ind_max = np.argmax(y_sown)

    for i in range (0,len(x_t[ind_max])):
        train_sample_xt = []
        train_sample_xa = []
        train_y = []
        print('for step {}'.format(i))
        for j in range (0,y_sown.shape[0]):
            if i < len(inputs[j]):
                train_sample_xt.append(inputs[j][i][0])
                train_sample_xa.append(inputs[j][i][1])
                train_y.append(y_sown[j])
        print('------------')
        print(len(train_sample_xt))
        print(len(train_sample_xa))
        print(train_sample_xt)
        print(train_sample_xa)
        print(train_y)
        print('------------')

        ###### linear regression ##########
        if len(train_y) > 3:
            train_sample_xt_array = np.expand_dims(np.asarray(train_sample_xt),axis=1)
            train_sample_xa_array = np.expand_dims(np.asarray(train_sample_xa),axis=1)
            X_feature = np.concatenate((train_sample_xt_array, train_sample_xa_array), axis=1)
            Y = train_y

            test_size = 3
            # Split the data into training/testing sets
            X_train = X_feature[:-test_size]
            X_test = X_feature[-test_size:]

            # Split the targets into training/testing sets
            y_train = Y[:-test_size]
            y_test = Y[-test_size:]

            # print(X_train.shape)
            # print(X_test.shape)
            # print(X_feature)
            # print(X_train)
            # print(X_test)

            reg = LinearRegression().fit(X_train, y_train)
            reg.score(X_train, y_train)

            # The coefficients
            print('Coefficients: \n', reg.coef_)
            y_pred = reg.predict(X_test)
            # The mean squared error
            print('Mean squared error: %.2f'
                % mean_squared_error(y_test, y_pred))
            print('R2: %.2f'
                % r2_score(y_test, y_pred))
            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination: %.2f'
                % r2_score(y_test, y_pred))














    # top_k_rain = []
    # top_p_rain = []

    # for i in range(0,chosen_crop.shape[0]):
    # # for i in range(0,1):
    #     doy_temp = chosen_crop.iloc[i, 1]

    #     rain_sub_temp = chosen_crop.iloc[i, 2]
    #     l = ast.literal_eval(rain_sub_temp)
    #     l_array = np.asarray(l)
    #     idx = (-l_array).argsort()[:1]

    #     rain_sub_accumulated_temp = chosen_crop.iloc[i, 3]
    #     l2 = ast.literal_eval(rain_sub_accumulated_temp)
    #     l2_array = np.asarray(l2)

    #     p=80 # my desired percentile, here 70% 
    #     # index of array entry nearest to percentile value
    #     pcen=np.percentile(l2_array,p,interpolation='nearest')
    #     i_near=abs(l2_array-pcen).argmin()

    #     top_p_rain.append(i_near)
    #     top_k_rain.append(idx[0])

    
    # print(top_p_rain)
    # print(top_k_rain)
    # print(chosen_crop['doy'].to_numpy())
    

    # top_p_rain_array = np.expand_dims(np.asarray(top_p_rain),axis=1)
    # top_k_rain_array = np.expand_dims(np.asarray(top_k_rain),axis=1)

    # X_feature = np.concatenate((top_p_rain_array, top_k_rain_array), axis=1)

    # Y = chosen_crop['doy'].to_numpy()


    # # test_size = 3
    # # # Split the data into training/testing sets
    # # X_train = X_feature[:-test_size]
    # # X_test = X_feature[-test_size:]

    # # # Split the targets into training/testing sets
    # # y_train = Y[:-test_size]
    # # y_test = Y[-test_size:]


    # # reg = LinearRegression().fit(X_train, y_train)
    # # reg.score(X_train, y_train)

    # # # The coefficients
    # # print('Coefficients: \n', reg.coef_)

    # # y_pred = reg.predict(X_test)

    # # # The mean squared error
    # # print('Mean squared error: %.2f'
    # #     % mean_squared_error(y_test, y_pred))
    # # print('R2: %.2f'
    # #     % r2_score(y_test, y_pred))
    # # # The coefficient of determination: 1 is perfect prediction
    # # print('Coefficient of determination: %.2f'
    # #     % r2_score(y_test, y_pred))


    # # # Plot outputs
    # # fig, ax = plt.subplots()
    # # ax.scatter(y_test, y_pred,  edgecolors=(0, 0, 0))
    # # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    # # ax.annotate("R2 = {:.3f}".format(r2_score(y_test, y_pred)), (90, 150),fontsize=25)
    # # ax.set_xlabel('Measured',fontsize=28)
    # # ax.set_ylabel('Predicted',fontsize=28)
    # # ax.set_xlim(80, 160)
    # # ax.set_ylim(80, 160)

    # # plt.show()





    # ################## plot rain, accumulated rain ##################
    # fig, host = plt.subplots()
    # par1 = host.twinx()

    # field_id = 3

    # rain_sub_temp = chosen_crop.iloc[field_id, 2]
    # l = ast.literal_eval(rain_sub_temp)
    # resampled = [sum(l[x:x+5]) for x in range(0, len(l), 5)]
    # ind = np.arange(len(resampled))


    # print(chosen_crop.iloc[field_id])




    # rain_sub_accumulated_temp = chosen_crop.iloc[field_id, 3]
    # l2 = ast.literal_eval(rain_sub_accumulated_temp)
    # resampled_2 = l2[::5]
    # ind2 = np.arange(len(resampled_2))   

    # p1 = host.bar(ind,resampled)
    # p2, = par1.plot(ind,resampled_2)

    # host.set_ylim(0, 100) 
    # par1.set_ylim(0, 300)
    # # host.set_xlabel("Date",fontsize=28)
    # host.set_ylabel("Rainfall (5 days sum)",fontsize=28)
    # par1.set_ylabel("Accumulated Rainfall",fontsize=28)

    # plt.axvline(x=91//5,color='red')
    # host.grid(False)
    # par1.grid(False)
    # plt.tight_layout() 
    # plt.show()








   