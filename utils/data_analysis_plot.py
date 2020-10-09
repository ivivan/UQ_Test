import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# sns.set(style='whitegrid')

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


def getdate_index(filepath, start, num_predict):
    """
    :param filepath: same dataset file
    :param start: start now no. for prediction
    :param num_predict: how many predictions
    :return: the x axis datatime index for prediciton drawing
    """
    dataset = pd.read_csv(filepath)
    dataset = dataset.iloc[start:start + num_predict, :]
    dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'], dayfirst=True)

    return dataset['TIMESTAMP']


# data input
filepath = 'data\Iowa\8_joined.csv'
dataset = pd.read_csv(filepath)
dataset['TIMESTAMP'] = pd.to_datetime(dataset['datetime'], dayfirst=True)
dataset.set_index(dataset['TIMESTAMP'],inplace=True)

df_choose = dataset.loc['2018-11-01':'2018-12-31'].copy()

print(df_choose.describe())

# Standlization, use StandardScaler
scaler_x = MinMaxScaler()
scaler_x.fit(
    df_choose[['temp_water_x', 'diss_oxy_con_x', 'nitrate_con_x', 'ph_x', 'spec_cond_x','temp_water_y', 'diss_oxy_con_y', 'nitrate_con_y', 'ph_y', 'spec_cond_y']])
df_choose[['temp_water_x', 'diss_oxy_con_x', 'nitrate_con_x', 'ph_x', 'spec_cond_x','temp_water_y', 'diss_oxy_con_y', 'nitrate_con_y', 'ph_y', 'spec_cond_y']] = scaler_x.transform(df_choose[['temp_water_x', 'diss_oxy_con_x', 'nitrate_con_x', 'ph_x', 'spec_cond_x','temp_water_y', 'diss_oxy_con_y', 'nitrate_con_y', 'ph_y', 'spec_cond_y']])




fig, ax = plt.subplots()

line1, = ax.plot(df_choose.index,df_choose['temp_water_x'],'b-',color=tableau20[5],label='Water Temperature')

line2, = ax.plot(df_choose.index,df_choose['diss_oxy_con_x'],'b-',color=tableau20[9],label='Dissolved Oxygen')

# line3, = ax.plot(df_choose.index,df_choose['nitrate_con_x'],'b-',color=tableau20[11],label='cc')

line4, = ax.plot(df_choose.index,df_choose['ph_x'],'b-',color=tableau20[7],label='pH')

# line5, = ax.plot(df_choose.index,df_choose['spec_cond_x'],'b-',color=tableau20[13],label='ee')

# plt.plot(df_choose.index,df_choose['temp_water_x'],'b-',color=tableau20[5],label='aa')

# plt.plot(df_choose.index,df_choose['ph_x'],color=tableau20[7])

# plt.plot(df_choose.index,df_choose['diss_oxy_con_x'],'b-',color=tableau20[9])

# plt.plot(df_choose.index,df_choose['nitrate_con_x'],'b-',color=tableau20[11])


ax.legend(fontsize=18)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Scaled Value',fontsize=18)

plt.gcf().autofmt_xdate()

plt.show()


















# fig, ((ax_one, ax_two, ax_three),(weekly_one, weekly_two, weekly_three),(daily_one, daily_two, daily_three)) = plt.subplots(3, 3, sharey=True,figsize=(9,9))


# ax_one.plot_date(dataset.index[0:48], dataset['DO_mg'][0:48], 'b-', color=tableau20[2])
# ax_two.plot_date(dataset.index[48:96], dataset['DO_mg'][48:96], 'b-', color=tableau20[2])
# ax_three.plot_date(dataset.index[96:144], dataset['DO_mg'][96:144], 'b-', color=tableau20[2])

# ax_one.get_xaxis().set_minor_locator(AutoMinorLocator())
# ax_one.grid(b=True, which='major', color='w', linewidth=1.5)
# ax_one.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(ax_one.get_xticklabels(), rotation=50, horizontalalignment='right')
# ax_one.set_ylabel('DO (mg)')

# ax_two.get_xaxis().set_minor_locator(AutoMinorLocator())
# ax_two.grid(b=True, which='major', color='w', linewidth=1.5)
# ax_two.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(ax_two.get_xticklabels(), rotation=50, horizontalalignment='right')

# ax_three.get_xaxis().set_minor_locator(AutoMinorLocator())
# ax_three.grid(b=True, which='major', color='w', linewidth=1.5)
# ax_three.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(ax_three.get_xticklabels(), rotation=50, horizontalalignment='right')


# weekly_one.plot_date(hourly_summary.index[0:168],hourly_summary['DO_mg'][0:168],'b-',color=tableau20[5])
# weekly_two.plot_date(hourly_summary.index[168:336],hourly_summary['DO_mg'][168:336],'b-',color=tableau20[5])
# weekly_three.plot_date(hourly_summary.index[336:504],hourly_summary['DO_mg'][336:504],'b-',color=tableau20[5])

# weekly_one.get_xaxis().set_minor_locator(AutoMinorLocator())
# weekly_one.grid(b=True, which='major', color='w', linewidth=1.5)
# weekly_one.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(weekly_one.get_xticklabels(), rotation=50, horizontalalignment='right')

# weekly_two.get_xaxis().set_minor_locator(AutoMinorLocator())
# weekly_two.grid(b=True, which='major', color='w', linewidth=1.5)
# weekly_two.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(weekly_two.get_xticklabels(), rotation=50, horizontalalignment='right')
# weekly_one.set_ylabel('DO (mg)')


# weekly_three.get_xaxis().set_minor_locator(AutoMinorLocator())
# weekly_three.grid(b=True, which='major', color='w', linewidth=1.5)
# weekly_three.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(weekly_three.get_xticklabels(), rotation=50, horizontalalignment='right')


# daily_one.plot_date(weekly_summary.index[0:30],weekly_summary['DO_mg'][0:30],'b-',color=tableau20[8])
# daily_two.plot_date(weekly_summary.index[30:61],weekly_summary['DO_mg'][30:61],'b-',color=tableau20[8])
# daily_three.plot_date(weekly_summary.index[61:92],weekly_summary['DO_mg'][61:92],'b-',color=tableau20[8])

# daily_one.get_xaxis().set_minor_locator(AutoMinorLocator())
# daily_one.grid(b=True, which='major', color='w', linewidth=1.5)
# daily_one.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(daily_one.get_xticklabels(), rotation=50, horizontalalignment='right')

# daily_two.get_xaxis().set_minor_locator(AutoMinorLocator())
# daily_two.grid(b=True, which='major', color='w', linewidth=1.5)
# daily_two.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(daily_two.get_xticklabels(), rotation=50, horizontalalignment='right')
# daily_one.set_ylabel('DO (mg)')


# daily_three.get_xaxis().set_minor_locator(AutoMinorLocator())
# daily_three.grid(b=True, which='major', color='w', linewidth=1.5)
# daily_three.grid(b=True, which='minor', color='w', linewidth=0.75)
# plt.setp(daily_three.get_xticklabels(), rotation=50, horizontalalignment='right')

# fig.tight_layout()
# fig.suptitle('DO Temporal Changing')
# fig.subplots_adjust(top=0.95)
# fig.show()
# plt.show()