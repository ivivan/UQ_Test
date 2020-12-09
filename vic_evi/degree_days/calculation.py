import numpy as np
from scipy.interpolate import interp1d


def calcDegreeDays(maxt, mint, crop):
    # calculate the constants to be used only included to see how they are calculated only need t_range_fract
    # calculate 3 hourly segments
    segments = np.arange(1, 9, 1)

    # calculate 3 hourly temperature
    t_range_fract = []

    for ss in segments:
        pt = 0.92105 + (0.1140*ss) - (0.0703*pow(ss, 2)) + \
            (0.0053 * pow(ss, 3))
        t_range_fract.append(pt)

    ######## different crop type ############
    ######### 'Barley' 'Canola' 'Lentils' 'Wheat' 'Chickpea'

    if crop == 'Barley':
        X = [0, 26, 34]
        Y = [0, 26, 0]
    elif crop == 'Canola':
        X = [0, 20, 35]
        Y = [0, 20, 0]
    elif crop == 'Lentils':
        X = [0, 30, 40]
        Y = [0, 30, 0]
    elif crop == 'Wheat':
        X = [0, 26, 34]
        Y = [0, 26, 0]
    elif crop == 'Chickpea':
        X = [0, 30, 40]
        Y = [0, 30, 0]

    ttp = interp1d(X, Y, kind='linear', fill_value='extrapolate')

    dd = maxt - mint
    p1 = ttp(mint + (dd*t_range_fract[0]))
    p2 = ttp(mint + (dd*t_range_fract[1]))
    p3 = ttp(mint + (dd*t_range_fract[2]))
    p4 = ttp(mint + (dd*t_range_fract[3]))
    p5 = ttp(mint + (dd*t_range_fract[4]))
    p6 = ttp(mint + (dd*t_range_fract[5]))
    p7 = ttp(mint + (dd*t_range_fract[6]))
    p8 = ttp(mint + (dd*t_range_fract[7]))

    df = np.array([p1, p2, p3, p4, p5, p6, p7, p8])

    tt = np.mean(df)

    return(tt)


def degreeDays_list(maxt_list, mint_list, crop):
    degreeD_list = []
    for i, j in zip(maxt_list, mint_list):
        degreeD_list.append(calcDegreeDays(i, j, crop))
    return degreeD_list


def all_degreeDays(maxt_list, mint_list, crop_list):

    data = {}

    for z in crop_list:
        degreeD_list = []
        for i, j in zip(maxt_list, mint_list):
            degreeD_list.append(calcDegreeDays(i, j, z))
        data[z] = degreeD_list
    return data


if __name__ == "__main__":

    maxt, mint = 25, 15
    crop = 'Canola'

    degreeDays = calcDegreeDays(maxt, mint, crop)

    print(degreeDays)
