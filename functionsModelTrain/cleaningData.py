import numpy as np
def balanced_hist_thresholding(b):
    i_s = np.min(np.where(b>0))# Starting point of histogram
    i_e = np.max(np.where(b>0))# End point of histogram
    i_m = (i_s + i_e)//2# Center of histogram
    w_l = np.sum(b[0:i_m+1])# Left side weight
    w_r = np.sum(b[i_m+1:i_e+1])# Right side weight


    while (i_s != i_e):# Until starting point not equal to endpoint
        if (w_r > w_l):# If right side is heavier
            w_r -= b[i_e]# Remove the end weight
            i_e -= 1

            if ((i_s+i_e)//2) < i_m:# Adjust the center position and recompute the weights
                w_l -= b[i_m]
                w_r += b[i_m]
                i_m -= 1
        else:# If left side is heavier, remove the starting weight
            w_l -= b[i_s]
            i_s += 1
            if ((i_s+i_e)//2) >= i_m:# Adjust the center position and recompute the weights
                w_l += b[i_m+1]
                w_r -= b[i_m+1]
                i_m += 1
    return i_m


def find_outliers(data):
    #Arrange the data in increasing order
    sorted(data)

    #Calculate first(q1) and third quartile(q3)
    q1, q3= np.percentile(data,[25,75])
    #print ("q1",q1, "q3",q3)

    #Find interquartile range (q3-q1)
    iqr = q3 - q1

    #Find lower bound q1*1.5
    #Find upper bound q3*1.5
    lower_bound = q1 -(1.5 * iqr)
    upper_bound = q3 +(1.5 * iqr)

    print "bounds:  ",lower_bound, "<->", upper_bound

    #Anything that lies outside of lower and upper bound is an outlier
    return lower_bound, upper_bound
