import numpy as np
from scipy.interpolate import interp1d


def calcDegreeDays(maxt, mint):
    # calculate the constants to be used only included to see how they are calculated only need t_range_fract
    # calculate 3 hourly segments
    segments = np.arange(1, 9, 1)

    # calculate 3 hourly temperature
    t_range_fract = []

    for ss in segments:
        pt = 0.92105 + (0.1140*ss) - (0.0703*pow(ss, 2)) + \
            (0.0053 * pow(ss, 3))
        t_range_fract.append(pt)

    # daily 3 hour interpolation THIS IS THE PLACE TO PUT IN CROP CARDINAL NUMBERS ****************************
    # 2 base

    # # taken from broccoli uses abase temp of 2 degrees >
    # X = [0,  2.0, 30.0, 35.0]  [0,26,34]
    # # taken from broccoli uses abase temp of 2 degrees > </y_tt>
    # Y = [0,  0.0, 28.0, 0.0]   [0,26,0]

    # taken from broccoli uses abase temp of 2 degrees >
    X = [0,26,34]
    # taken from broccoli uses abase temp of 2 degrees > </y_tt>
    Y = [0,26,0]

    ttp = interp1d(X, Y, kind='linear',fill_value='extrapolate')

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


def degreeDays_list(maxt_list, mint_list):
    degreeD_list = []
    for i,j in zip(maxt_list,mint_list):
        degreeD_list.append(calcDegreeDays(i,j))
    return degreeD_list

if __name__ == "__main__":

    maxt, mint = 25, 15

    degreeDays = calcDegreeDays(maxt, mint)

    print(degreeDays)
