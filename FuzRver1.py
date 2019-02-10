"""
FuzR
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
Since FuzR works only in the range of 0-1o, using min-max normalization it was brought into the range
and the factor wa calculated
2. Distribution 2 - [0-10]
FuzR factor was calculated directly for this distribution as it is already in the required range

TODO
1. The code needs to become more reconfigurable to take inputs as parameters and modify the code throughout
using variables rather than manually calculating the values for a predefined distribution
2. Analysis
"""
from mfArea import mfAreaCertainity as mfAC
from mfArea import mfAreaUncertainity as mfAU
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from scipy.stats import norm
from mfParameters import*
"""
Distribution 1 [0-100]
--------------
Intervals | Lower Limit | Upper Limit |
Interval1 |    0        |    28       |
Interval2 |    26       |    40       |
Interval3 |    36       |    52       |
Interval4 |    44       |    71       |
Interval5 |    63       |    94       |

Distribution 2 [0-10]
--------------
Intervals | Lower Limit | Upper Limit |
Interval1 |    0        |    1        |
Interval2 |    0.5      |    3.2      |
Interval3 |    2.4      |    6        |
Interval4 |    5        |    10       |
"""
dist1 = np.arange(0, 101, .01)
dist2 = np.arange(0, 11, .01)

# ***********************TRIANGULAR MF'S - Distribution 1*****************************
mft1d1 = fuzz.trimf(dist1, [0.0, 14.0, 28.0])
mft2d1 = fuzz.trimf(dist1, [26.0, 33.0, 40.0])
mft3d1 = fuzz.trimf(dist1, [36.0, 44.0, 52.0])
mft4d1 = fuzz.trimf(dist1, [44.0, 57.5, 71.0])
mft5d1 = fuzz.trimf(dist1, [63.0, 78.5, 94.0])

# ***********************TRIANGULAR MF'S - Distribution 2*****************************
mft1d2 = fuzz.trimf(dist2, [0.0, 0.5, 1.0])
mft2d2 = fuzz.trimf(dist2, [0.5, 1.85, 3.2])
mft3d2 = fuzz.trimf(dist2, [2.4, 4.2, 6.0])
mft4d2 = fuzz.trimf(dist2, [5.0, 7.5, 10.0])

fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(7, 9))

ax0.plot(dist1, mft1d1, 'b', linewidth=1)
ax0.plot(dist1, mft2d1, 'b', linewidth=1)
ax0.plot(dist1, mft3d1, 'b', linewidth=1)
ax0.plot(dist1, mft4d1, 'b', linewidth=1)
ax0.plot(dist1, mft5d1, 'b', linewidth=1)

ax1.plot(dist2, mft1d2, 'g', linewidth=1)
ax1.plot(dist2, mft2d2, 'g', linewidth=1)
ax1.plot(dist2, mft3d2, 'g', linewidth=1)
ax1.plot(dist2, mft4d2, 'g', linewidth=1)

# Area of Uncertainity - Dist1
a1tUd1 = mfAU.trimfAreaU(dist1, [0.0, 14.0, 28.0], [26.0, 33.0, 40.0])
a2tUd1 = mfAU.trimfAreaU(dist1, [26.0, 33.0, 40.0], [36.0, 44.0, 52.0])
a3tUd1 = mfAU.trimfAreaU(dist1, [36.0, 44.0, 52.0], [44.0, 57.5, 71.0])
a4tUd1 = mfAU.trimfAreaU(dist1, [44.0, 57.5, 71.0], [63.0, 78.5, 94.0])

# Area of Uncertainity - Dist2
a1tUd2 = mfAU.trimfAreaU(dist2, [0.0, 0.5, 1.0], [0.5, 1.85, 3.2])
a2tUd2 = mfAU.trimfAreaU(dist2, [0.5, 1.85, 3.2], [2.4, 4.2, 6.0])
a3tUd2 = mfAU.trimfAreaU(dist2, [2.4, 4.2, 6.0], [5.0, 7.5, 10.0])
print("\n****************UNCERTAINITY - Distribution 1 - Triangular MF's**************************")
print("AreaU 1 = {}".format(a1tUd1))
print("AreaU 2 = {}".format(a2tUd1))
print("AreaU 3 = {}".format(a3tUd1))
print("AreaU 4 = {}".format(a4tUd1))
totAreaUd1 = a1tUd1 + a2tUd1 + a3tUd1 + a4tUd1
print("Total Area of Uncertainity = {}".format(totAreaUd1))

print("\n****************UNCERTAINITY - Distribution 2 - Triangular MF's**************************")
print("AreaU 1 = {}".format(a1tUd2))
print("AreaU 2 = {}".format(a2tUd2))
print("AreaU 3 = {}".format(a3tUd2))
totAreaUd2 = a1tUd2 + a2tUd2 + a3tUd2
print("Total Area of Uncertainity = {}".format(totAreaUd2))
"""
# Area of Certainity - Dist1
a1tCd1 = mfAC.trimfAreaC(dist1, [0.0, 14.0, 28.0], 0, a1tUd1)
a2tCd1 = mfAC.trimfAreaC(dist1, [26.0, 33.0, 40.0], a1tUd1, a2tUd1)
a3tCd1 = mfAC.trimfAreaC(dist1, [36.0, 44.0, 52.0], a2tUd1, a3tUd1)
a4tCd1 = mfAC.trimfAreaC(dist1, [44.0, 57.5, 71.0], a3tUd1, a4tUd1)
a5tCd1 = mfAC.trimfAreaC(dist1, [63.0, 78.5, 94.0], a4tUd1, 0)
"""
# Area of Certainity - Dist2
a1tCd2 = mfAC.trimfAreaC(dist2, [0.0, 0.5, 1.0], 0, a1tUd2)
a2tCd2 = mfAC.trimfAreaC(dist2, [0.5, 1.85, 3.2], a1tUd2, a2tUd2)
a3tCd2 = mfAC.trimfAreaC(dist2, [2.4, 4.2, 6.0], a2tUd2, a3tUd2)
a4tCd2 = mfAC.trimfAreaC(dist2, [5.0, 7.5, 10.0], a3tUd2, 0)
"""
print("******************CERTAINITY - Distribution 1 - Triangular MF's**************************")
print("AreaU 1 = {}".format(a1tCd1))
print("AreaU 2 = {}".format(a2tCd1))
print("AreaU 3 = {}".format(a3tCd1))
print("AreaU 4 = {}".format(a4tCd1))
print("AreaU 5 = {}".format(a5tCd1))
totAreaCd1 = a1tCd1 + a2tCd1 + a3tCd1 + a4tCd1 + a5tCd1
print("Total Area of Certainity = {}".format(totAreaCd1))
"""
print("******************CERTAINITY - Distribution 2 - Triangular MF's**************************")
print("AreaU 1 = {}".format(a1tCd2))
print("AreaU 2 = {}".format(a2tCd2))
print("AreaU 3 = {}".format(a3tCd2))
print("AreaU 4 = {}".format(a4tCd2))
totAreaCd2 = a1tCd2 + a2tCd2 + a3tCd2 + a4tCd2
print("Total Area of Certainity = {}".format(totAreaCd2))

#print("\nfactor for TRIANGULAR MEMBERSHIP FUNCTION- Distribution 1 : {}".format(totAreaCd1/totAreaUd1))

print("\nfactor for TRIANGULAR MEMBERSHIP FUNCTION- Distribution 2 : {}".format(totAreaCd2/totAreaUd2))

# ***********************GAUSSIAN MF'S****************************
"""
Distribution 1 [0-100]
--------------
Intervals | Lower Limit | Upper Limit |
Interval1 |    0        |    28       |
Interval2 |    26       |    40       |
Interval3 |    36       |    52       |
Interval4 |    44       |    71       |
Interval5 |    63       |    94       |

Distribution 2 [0-10]
--------------
Intervals | Lower Limit | Upper Limit |
Interval1 |    0        |    1        |
Interval2 |    0.5      |    3.2      |
Interval3 |    2.4      |    6        |
Interval4 |    5        |    10       |
"""

# ***********************GAUSSIAN MF'S - Distribution 1*****************************
mfg1d1 = fuzz.gaussmf(dist1, 14, 4.66)  # 65.3
mfg2d1 = fuzz.gaussmf(dist1, 33, 2.33)  # 16.3
mfg3d1 = fuzz.gaussmf(dist1, 44, 21.3)
mfg4d1 = fuzz.gaussmf(dist1, 57.5, 24.1)
mfg5d1 = fuzz.gaussmf(dist1, 78.5, 36.7)

# ***********************GAUSSIAN MF'S - Distribution 2*****************************
mfg1d2 = fuzz.gaussmf(dist2, 0.5, 0.083)
mfg2d2 = fuzz.gaussmf(dist2, 1.85, 0.6)
mfg3d2 = fuzz.gaussmf(dist2, 4.2, 1.08)
mfg4d2 = fuzz.gaussmf(dist2, 7.5, 2.08)

ax2.plot(dist1, mfg1d1, 'r', linewidth=1)
ax2.plot(dist1, mfg2d1, 'r', linewidth=1)
ax2.plot(dist1, mfg3d1, 'r', linewidth=1)
ax2.plot(dist1, mfg4d1, 'r', linewidth=1)
ax2.plot(dist1, mfg5d1, 'r', linewidth=1)

ax3.plot(dist2, mfg1d2, 'r', linewidth=1)
ax3.plot(dist2, mfg2d2, 'r', linewidth=1)
ax3.plot(dist2, mfg3d2, 'r', linewidth=1)
ax3.plot(dist2, mfg4d2, 'r', linewidth=1)

# Area of Uncertainity - Dist1
a1gUd1 = mfAU.gaussmfAreaU(dist1, [14, 65.3], [33, 16.3])
a2gUd1 = mfAU.gaussmfAreaU(dist1, [33, 16.3], [44, 21.3])
a3gUd1 = mfAU.gaussmfAreaU(dist1, [44, 21.3], [57.5, 24.1])
a4gUd1 = mfAU.gaussmfAreaU(dist1, [57.5, 24.1], [78.5, 36.7])

# Area of Uncertainity - Dist2
a1gUd2 = mfAU.gaussmfAreaU(dist2, [0.5, 0.083], [1.85, 0.6])
a2gUd2 = mfAU.gaussmfAreaU(dist2, [1.85, 0.6], [4.2, 1.08])
a3gUd2 = mfAU.gaussmfAreaU(dist2, [4.2, 1.08], [7.5, 2.08])

print("\n****************UNCERTAINITY - Distribution 1 - GAUSSIAN MF's**************************")
print("AreaU 1 = {} ".format(a1gUd1))
print("AreaU 2 = {} ".format(a2gUd1))
print("AreaU 3 = {} ".format(a3gUd1))
print("AreaU 4 = {} ".format(a4gUd1))
totAreagUd1 = a1gUd1 + a2gUd1 + a3gUd1 + a4gUd1
print("Total Area of Uncertainity = {}".format(totAreagUd1))

print("\n****************UNCERTAINITY - Distribution 2 - GAUSSIAN MF's**************************")
print("AreaU 1 = {} ".format(a1gUd2))
print("AreaU 2 = {} ".format(a2gUd2))
print("AreaU 3 = {} ".format(a3gUd2))
totAreagUd2 = a1gUd2 + a2gUd2 + a3gUd2
print("Total Area of Uncertainity = {}".format(totAreagUd2))

# Area of Certainity - Dist1
a1gCd1 = mfAC.gaussmfAreaC(dist1, [14, 65.3], 0, a1gUd1)
a2gCd1 = mfAC.gaussmfAreaC(dist1, [33, 16.3], a1gUd1, a2gUd1)
a3gCd1 = mfAC.gaussmfAreaC(dist1, [44, 21.3], a2gUd1, a3gUd1)
a4gCd1 = mfAC.gaussmfAreaC(dist1, [57.5, 24.1], a3gUd1, a4gUd1)
a5gCd1 = mfAC.gaussmfAreaC(dist1, [78.5, 36.7], a4gUd1, 0)

# Area of Certainity - Dist2
a1gCd2 = mfAC.gaussmfAreaC(dist2, [0.5, 0.083], 0, a1gUd2)
a2gCd2 = mfAC.gaussmfAreaC(dist2, [1.85, 0.6], a1gUd2, a2gUd2)
a3gCd2 = mfAC.gaussmfAreaC(dist2, [4.2, 1.08], a2gUd2, a3gUd2)
a4gCd2 = mfAC.gaussmfAreaC(dist2, [7.5, 2.08], a3gUd2, 0)

print("******************CERTAINITY - Distribution 1 - GAUSSIAN MF's**************************")
print("AreaU 1 = {} ".format(a1gCd1))
print("AreaU 2 = {} ".format(a2gCd1))
print("AreaU 3 = {} ".format(a3gCd1))
print("AreaU 4 = {} ".format(a4gCd1))
print("AreaU 5 = {} ".format(a5gCd1))
totAreagCd1 = a1gCd1 + a2gCd1 + a3gCd1 + a4gCd1 + a5gCd1
print("Total Area of Uncertainity = {}".format(totAreagCd1))

print("******************CERTAINITY - Distribution 2 - GAUSSIAN MF's**************************")
print("AreaU 1 = {} ".format(a1gCd2))
print("AreaU 2 = {} ".format(a2gCd2))
print("AreaU 3 = {} ".format(a3gCd2))
print("AreaU 4 = {} ".format(a4gCd2))
totAreagCd2 = a1gCd2 + a2gCd2 + a3gCd2 + a4gCd2
print("Total Area of Uncertainity = {}".format(totAreagCd2))

print("\nfactor for GAUSSIAN MEMBERSHIP FUNCTION- Distribution 1 : {}".format(totAreagCd1/totAreagUd1))

print("\nfactor for GAUSSIAN MEMBERSHIP FUNCTION- Distribution 2 : {}".format(totAreagCd2/totAreagUd2))

# Show plot
# plt.tight_layout()

# ***********************TRAPEZOIDAL MF'S*****************************
"""
Distribution 1 [0-100]
--------------
Intervals | Lower Limit | Upper Limit |
Interval1 |    0        |    28       |
Interval2 |    26       |    40       |
Interval3 |    36       |    52       |
Interval4 |    44       |    71       |
Interval5 |    63       |    94       |

Distribution 2 [0-10]
--------------
Intervals | Lower Limit | Upper Limit |
Interval1 |    0        |    1        |
Interval2 |    0.5      |    3.2      |
Interval3 |    2.4      |    6        |
Interval4 |    5        |    10       |
"""
# ***********************TRAPEZOIDAL MF'S - Distribution 1*********************
mftp1d1 = fuzz.trapmf(dist1, [0.0, 7.0, 21.0, 28.0])
mftp2d1 = fuzz.trapmf(dist1, [26.0, 29.5, 36.5, 40.0])
mftp3d1 = fuzz.trapmf(dist1, [36.0, 40.0, 48.0, 52.0])
mftp4d1 = fuzz.trapmf(dist1, [44.0, 50.7, 64.1, 71.0])
mftp5d1 = fuzz.trapmf(dist1, [63.0, 70.7, 83.1, 94.0])

# ***********************TRAPEZOIDAL MF'S - Distribution 2********************
mftp1d2 = fuzz.trapmf(dist2, [0.0, 0.25, 0.75, 1.0])
mftp2d2 = fuzz.trapmf(dist2, [0.5, 1.17, 2.51, 3.2])
mftp3d2 = fuzz.trapmf(dist2, [2.4, 3.3, 5.1, 6])
mftp4d2 = fuzz.trapmf(dist2, [5, 6.25, 8.75, 10])

ax4.plot(dist1, mftp1d1, 'b', linewidth=1)
ax4.plot(dist1, mftp2d1, 'b', linewidth=1)
ax4.plot(dist1, mftp3d1, 'b', linewidth=1)
ax4.plot(dist1, mftp4d1, 'b', linewidth=1)
ax4.plot(dist1, mftp5d1, 'b', linewidth=1)

ax5.plot(dist2, mftp1d2, 'g', linewidth=1)
ax5.plot(dist2, mftp2d2, 'g', linewidth=1)
ax5.plot(dist2, mftp3d2, 'g', linewidth=1)
ax5.plot(dist2, mftp4d2, 'g', linewidth=1)

# Area of Uncertainity - Dist1
a1tpUd1 = mfAU.trapmfAreaU(dist1, [0.0, 7.0, 21.0, 28.0], [26.0, 29.5, 36.5, 40.0])
a2tpUd1 = mfAU.trapmfAreaU(dist1, [26.0, 29.5, 36.5, 40.0], [36.0, 40.0, 48.0, 52.0])
a3tpUd1 = mfAU.trapmfAreaU(dist1, [36.0, 40.0, 48.0, 52.0], [44.0, 50.7, 64.1, 71.0])
a4tpUd1 = mfAU.trapmfAreaU(dist1, [44.0, 50.7, 64.1, 71.0], [63.0, 70.7, 83.1, 94.0])

# Area of Uncertainity - Dist2
a1tpUd2 = mfAU.trapmfAreaU(dist2, [0.0, 0.25, 0.75, 1.0], [0.5, 1.17, 2.51, 3.2])
a2tpUd2 = mfAU.trapmfAreaU(dist2, [0.5, 1.17, 2.51, 3.2], [2.4, 3.3, 5.1, 6])
a3tpUd2 = mfAU.trapmfAreaU(dist2, [2.4, 3.3, 5.1, 6], [5, 6.25, 8.75, 10])
print("\n****************UNCERTAINITY - Distribution 1 - TRAPEZOIDAL MF's**************************")
print("AreaU 1 = ")
print(a1tpUd1)
print("AreaU 2 = ")
print(a2tpUd1)
print("AreaU 3 = ")
print(a3tpUd1)
print("AreaU 4 = ")
print(a4tpUd1)
print("Total Area of Uncertainity =")
totAreatpUd1 = a1tpUd1 + a2tpUd1 + a3tpUd1 + a4tpUd1
print(totAreatpUd1)

print("\n****************UNCERTAINITY - Distribution 2 - TRAPEZOIDAL MF's**************************")
print("AreaU 1 = ")
print(a1tpUd2)
print("AreaU 2 = ")
print(a2tpUd2)
print("AreaU 3 = ")
print(a3tpUd2)
print("Total Area of Uncertainity =")
totAreatpUd2 = a1tpUd2 + a2tpUd2 + a3tpUd2
print(totAreatpUd2)

# Area of Certainity - Dist1
a1tpCd1 = mfAC.trapmfAreaC(dist1, [0.0, 7.0, 21.0, 28.0], [26.0, 29.5, 36.5, 40.0], 0, a1tpUd1)
a2tpCd1 = mfAC.trapmfAreaC(dist1, [26.0, 29.5, 36.5, 40.0], [36.0, 40.0, 48.0, 52.0], a1tpUd1, a2tpUd1)
a3tpCd1 = mfAC.trapmfAreaC(dist1, [36.0, 40.0, 48.0, 52.0], [44.0, 50.7, 64.1, 71.0], a2tpUd1, a3tpUd1)
a4tpCd1 = mfAC.trapmfAreaC(dist1, [44.0, 50.7, 64.1, 71.0], [ 63.0, 70.7, 83.1, 94.0], a3tpUd1, a4tpUd1)
a5tpCd1 = mfAC.trapmfAreaC(dist1, [63.0, 70.7, 83.1, 94.0], [0, 0, 0, 0], a4tpUd1, 0)

# Area of Certainity - Dist2
a1tpCd2 = mfAC.trapmfAreaC(dist2, [0.0, 0.25, 0.75, 1.0], [0.5, 1.17, 2.51, 3.2], 0, a1tpUd2)
a2tpCd2 = mfAC.trapmfAreaC(dist2, [0.5, 1.17, 2.51, 3.2], [2.4, 3.3, 5.1, 6], a1tpUd2, a2tpUd2)
a3tpCd2 = mfAC.trapmfAreaC(dist2, [2.4, 3.3, 5.1, 6], [5, 6.25, 8.75, 10], a2tpUd2, a3tpUd2)

a4tpCd2 = mfAC.trapmfAreaC(dist2, [5, 6.25, 8.75, 10], [0, 0, 0, 0], a3tpUd2, 0)
print("******************CERTAINITY - Distribution 1 - TRAPEZOIDAL MF's**************************")
print("AreaU 1 = ")
print(a1tpCd1)
print("AreaU 2 = ")
print(a2tpCd1)
print("AreaU 3 = ")
print(a3tpCd1)
print("AreaU 4 = ")
print(a4tpCd1)
print("AreaU 5 = ")
print(a5tpCd1)
print("Total Area of Certainity =")
totAreatpCd1 = a1tpCd1 + a2tpCd1 + a3tpCd1 + a4tpCd1 + a5tpCd1
print(totAreatpCd1)

print("******************CERTAINITY - Distribution 2 - TRAPEZOIDAL MF's**************************")
print("AreaU 1 = ")
print(a1tpCd2)
print("AreaU 2 = ")
print(a2tpCd2)
print("AreaU 3 = ")
print(a3tpCd2)
print("AreaU 4 = ")
print(a4tpCd2)
print("Total Area of Certainity =")
totAreatpCd2 = a1tpCd2 + a2tpCd2 + a3tpCd2 + a4tpCd2
print(totAreatpCd2)

print("\nfactor for TRAPEZOIDAL MEMBERSHIP FUNCTION- Distribution 1")
print(totAreatpCd1/totAreatpUd1)

print("\nfactor for TRAPEZOIDAL MEMBERSHIP FUNCTION- Distribution 2")
print(totAreatpCd2/totAreatpUd2)


"""
# Test & Validation

t1 = fuzz.trimf(dist2, [0.0, 2.0, 4.0])
t2 = fuzz.trimf(dist2, [2.0, 4.0, 6.0])
t3 = fuzz.trimf(dist2, [3.0, 6.0, 9.0])
t4 = fuzz.trimf(dist2, [6.0, 8.0, 10.0])

g1 = fuzz.gaussmf(dist2, 2, 0.75)
g2 = fuzz.gaussmf(dist2, 4, 0.75)
g3 = fuzz.gaussmf(dist2, 6, 1.5)
g4 = fuzz.gaussmf(dist2, 8, 0.75)

ax2.plot(dist2, t1, 'g', linewidth=1)
ax2.plot(dist2, t2, 'g', linewidth=1)
ax2.plot(dist2, t3, 'g', linewidth=1)
ax2.plot(dist2, t4, 'g', linewidth=1)

ax3.plot(dist2, g1, 'r', linewidth=1)
ax3.plot(dist2, g2, 'r', linewidth=1)
ax3.plot(dist2, g3, 'r', linewidth=1)
ax3.plot(dist2, g4, 'r', linewidth=1)
"""
