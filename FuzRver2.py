"""
FuzR - Ver2 [more Object oriented, more pythonic and accepts user defined intervals as inputs]
------
FuzR is an approach for efficient and optimal deisgn of fuzzy systems without the need of expert knowledge or an
an expert to know how to automatically update the parameters or choose the membership functions for the
system. This approach utilizes the filter approach which less computationally intensive compared to the
wrapper approach to modify or self adapt the parameter by using evolutionary algorithms or neural networks

FuzR is the ratio between the area of certainity and the area of uncertainty
FuzR only works on datasets in the range 0-10, hence for any datasets with a much larger range, min-max
normalization must be applied to bring it to that range.

Below, this code represents the FuzR factor for gaussian, trapezoidal and triangular membership functions
for a given predefined set of intervals with the lower and upper limit.
The factor was determined for two distributions namely:
1. Distribution 1 -  [0-100]
Since FuzR works only in the range of 0-10, using min-max normalization it was brought into the range
and the factor wa calculated
2. Distribution 2 - [0-10]
FuzR factor was calculated directly for this distribution as it is already in the required range

TODO
1. Analysis
2. Research Scaling Factor
"""
from mfArea import mfAreaCertainity as mfAC
from mfArea import mfAreaUncertainity as mfAU
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from scipy.stats import norm
from mfParameters import*
from sympy import*

dist1 = np.arange(0, 10, .01)
#dist2 = np.arange(0, 115, .01)

# Define Intervals
intervals = []
n = int(input("Enter the number of intervals : "))
for i in range(n):
    inter = input()
    intervals.append(list(map(float, inter.split(","))))
l = len(intervals)
# Convert large distributions to 0-10 range


# Define Parameters
param = parameters(intervals)
intervalsMod = param.checkChange()
for i in range(len(intervalsMod)):
    print("intervalsMod[i]")
paramMod = parameters(intervalsMod)
triParam = paramMod.triParam()
trapParam = paramMod.trapParam()
gaussParam = paramMod.gaussParam()
print("Gaussian Parameters")
print(gaussParam)

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

fig1, ax0 = plt.subplots(nrows=1, figsize=(6, 6))
fig1, ax1 = plt.subplots(nrows=1, figsize=(6, 6))
fig1, ax2 = plt.subplots(nrows=1, figsize=(6, 6))
ax0.spines['top'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
# TRIANGULAR MF'S
triMFSet = []
for i in range(l):
    print(triParam[i])
    triMFSet.append(fuzz.trimf(dist1, triParam[i]))
    ax0.plot(dist1, triMFSet[i], 'k', linewidth = 1)

# TRAPEZOIDAL MF'S
trapMFSet = []
for i in range(l):
    print(trapParam[i])
    trapMFSet.append(fuzz.trapmf(dist1, trapParam[i]))
    ax1.plot(dist1, trapMFSet[i], 'k', linewidth = 1)


# GAUSSIAN MF'S
gaussMFSet = []
for i in range(l):
    print(gaussParam[i])
    gaussMFSet.append(fuzz.gaussmf(dist1, gaussParam[i][0], gaussParam[i][1]))
    ax2.plot(dist1, gaussMFSet[i], 'k', linewidth = 1)

# TRIANGULAR MF'S - UNCERTAINITY
triUA = []
triTotUA = 0
print("\nTRIANGULAR - AREA OF UNCERTAINITY")
for i in range(l-1):
    triUA.append(mfAU.trimfAreaU(dist1, triParam[i], triParam[i+1]))
    print("AreaUA {} : {} ".format(i+1, triUA[i]))
    triTotUA = triTotUA + triUA[i]
print("Total Uncertainity Area: {}".format(triTotUA))

# TRIANGULAR MF'S - CERTAINITY
triCA = []
triTotCA = 0
print("\nTRIANGULAR - AREA OF CERTAINITY")
for i in range(l):
    if(i==0):
        triCA.append(mfAC.trimfAreaC(dist1, triParam[0], 0, triUA[0]))
    elif(i==(l-1)):
        #print("TEST 1 : {}".format(mfAC.trimfAreaC(dist1, triParam[l-1], triUA[i-1], 0)))
        #print("TEST 2 : {}".format( mfAC.trimfAreaC(dist1, [5.0, 7.5, 10.0], 0.116, 0)))
        triCA.append(mfAC.trimfAreaC(dist1, triParam[l-1], triUA[i-1], 0))
    else:
        triCA.append(mfAC.trimfAreaC(dist1, triParam[i], triUA[i-1], triUA[i]))
    print("AreaCA {} : {} ".format(i+1, triCA[i]))
    triTotCA = triTotCA + triCA[i]
print("Total Certainity Area: {}".format(triTotCA))

# TRIANGULAR MF'S - FuzR MEASSURE
FuzRtri = triTotCA/triTotUA
print("FuzR Meassure - Triangular : {}".format(FuzRtri))

# TRAPEZOIDAL MF'S - UNCERTAINITY
trapUA = []
trapTotUA = 0
print("\nTRAPEZOIDAL - AREA OF UNCERTAINITY")
for i in range(l-1):
    trapUA.append(mfAU.trapmfAreaU(dist1, trapParam[i], trapParam[i+1]))
    print("AreaUA {} : {} ".format(i+1, trapUA[i]))
    trapTotUA = trapTotUA + trapUA[i]
print("Total Uncertainity Area: {}".format(trapTotUA))

# TRAPEZOIDAL MF'S - CERTAINITY
trapCA = []
trapTotCA = 0
print("\nTRAPEZOIDAL - AREA OF CERTAINITY")
for i in range(l):
    if(i==0):
        trapCA.append(mfAC.trapmfAreaC(dist1, trapParam[0], trapParam[1], 0, trapUA[0]))
    elif(i==(l-1)):
        trapCA.append(mfAC.trapmfAreaC(dist1, trapParam[l-1], [0, 0, 0, 0], trapUA[i-1], 0))
    else:
        trapCA.append(mfAC.trapmfAreaC(dist1, trapParam[i], trapParam[i+1], trapUA[i-1], trapUA[i]))
    print("AreaCA {} : {} ".format(i+1, trapCA[i]))
    trapTotCA = trapTotCA + trapCA[i]
print("Total Certainity Area: {}".format(trapTotCA))

# TRAPEZOIDAL MF'S - FuzR MEASSURE
FuzRtrap = trapTotCA/trapTotUA
print("FuzR Meassure - Trapezoidal : {}".format(FuzRtrap))

# GAUSSIAN MF'S - UNCERTAINITY
gaussUA = []
gaussTotUA = 0
print("\nGAUSSIAN - AREA OF UNCERTAINITY")
for i in range(l-1):
    gaussUA.append(mfAU.gaussmfAreaU(dist1, gaussParam[i], gaussParam[i+1]))
    print("AreaUA {} : {} ".format(i+1, gaussUA[i]))
    gaussTotUA = gaussTotUA + gaussUA[i]
print("Total Uncertainity Area: {}".format(gaussTotUA))

# GAUSSIAN MF'S - CERTAINITY
gaussCA = []
gaussTotCA = 0
print("\nGAUSSIAN - AREA OF CERTAINITY")
for i in range(l):
    if(i==0):
        gaussCA.append(mfAC.gaussmfAreaC(dist1, gaussParam[0], gaussParam[1], 0, gaussUA[0]))
    elif(i==(l-1)):
        gaussCA.append(mfAC.gaussmfAreaC(dist1, gaussParam[l-1], [0, 0], gaussUA[i-1], 0))
    else:
        gaussCA.append(mfAC.gaussmfAreaC(dist1, gaussParam[i], gaussParam[i+1], gaussUA[i-1], gaussUA[i]))
    print("AreaCA {} : {} ".format(i+1, gaussCA[i]))
    gaussTotCA = gaussTotCA + gaussCA[i]
print("Total Certainity Area: {}".format(gaussTotCA))

# GAUSSIAN MF'S - FuzR MEASSURE
FuzRgauss = gaussTotCA/gaussTotUA
print("FuzR Meassure - GAUSSIAN : {}".format(FuzRgauss))

#plt.margins(0)
plt.tight_layout()
plt.show()
