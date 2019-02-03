from mfArea import mfAreaCertainity as mfAC
from mfArea import mfAreaUncertainity as mfAU
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from scipy.stats import norm
# data = pd.read_excel("data.xlsx")
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

fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(7, 10))
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
print("AreaU 1 = ")
print(a1tUd1)
print("AreaU 2 = ")
print(a2tUd1)
print("AreaU 3 = ")
print(a3tUd1)
print("AreaU 4 = ")
print(a4tUd1)
print("Total Area of Uncertainity =")
totAreaUd1 = a1tUd1 + a2tUd1 + a3tUd1 + a4tUd1
print(totAreaUd1)

print("\n****************UNCERTAINITY - Distribution 2 - Triangular MF's**************************")
print("AreaU 1 = ")
print(a1tUd2)
print("AreaU 2 = ")
print(a2tUd2)
print("AreaU 3 = ")
print(a3tUd2)
print("Total Area of Uncertainity =")
totAreaUd2 = a1tUd2 + a2tUd2 + a3tUd2
print(totAreaUd2)

# Area of Certainity - Dist1
a1tCd1 = mfAC.trimfAreaC(dist1, [0.0, 14.0, 28.0], [26.0, 33.0, 40.0], 0, a1tUd1)
a2tCd1 = mfAC.trimfAreaC(dist1, [26.0, 33.0, 40.0], [36.0, 44.0, 52.0], a1tUd1, a2tUd1)
a3tCd1 = mfAC.trimfAreaC(dist1, [36.0, 44.0, 52.0], [44.0, 57.5, 71.0], a2tUd1, a3tUd1)
a4tCd1 = mfAC.trimfAreaC(dist1, [44.0, 57.5, 71.0], [63.0, 78.5, 94.0], a3tUd1, a4tUd1)
a5tCd1 = mfAC.trimfAreaC(dist1, [63.0, 78.5, 94.0], [0, 0, 0], a4tUd1, 0)

# Area of Certainity - Dist2
a1tCd2 = mfAC.trimfAreaC(dist2, [0.0, 0.5, 1.0], [0.5, 1.85, 3.2], 0, a1tUd2)
a2tCd2 = mfAC.trimfAreaC(dist2, [0.5, 1.85, 3.2], [2.4, 4.2, 6.0], a1tUd2, a2tUd2)
a3tCd2 = mfAC.trimfAreaC(dist2, [2.4, 4.2, 6.0], [5.0, 7.5, 10.0], a2tUd2, a3tUd2)
a4tCd2 = mfAC.trimfAreaC(dist2, [5.0, 7.5, 10.0], [0, 0, 0], a3tUd1, 0)
print("******************CERTAINITY - Distribution 1 - Triangular MF's**************************")
print("AreaU 1 = ")
print(a1tCd1)
print("AreaU 2 = ")
print(a2tCd1)
print("AreaU 3 = ")
print(a3tCd1)
print("AreaU 4 = ")
print(a4tCd1)
print("AreaU 5 = ")
print(a5tCd1)
print("Total Area of Certainity =")
totAreaCd1 = a1tCd1 + a2tCd1 + a3tCd1 + a4tCd1 + a5tCd1
print(totAreaCd1)

print("******************CERTAINITY - Distribution 2 - Triangular MF's**************************")
print("AreaU 1 = ")
print(a1tCd2)
print("AreaU 2 = ")
print(a2tCd2)
print("AreaU 3 = ")
print(a3tCd2)
print("AreaU 4 = ")
print(a4tCd2)
print("Total Area of Certainity =")
totAreaCd2 = a1tCd2 + a2tCd2 + a3tCd2 + a4tCd2
print(totAreaCd2)

print("\nfactor for TRIANGULAR MEMBERSHIP FUNCTION- Distribution 1")
print(totAreaCd1/totAreaUd1)

print("\nfactor for TRIANGULAR MEMBERSHIP FUNCTION- Distribution 2")
print(totAreaCd2/totAreaUd2)

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
mfg1d1 =fuzz.gaussmf(dist1, 14, 65.3)
mfg2d1 = fuzz.gaussmf(dist1, 33, 16.3)
mfg3d1 = fuzz.gaussmf(dist1, 44, 21.3)
mfg4d1 = fuzz.gaussmf(dist1, 57.5, 24.1)
mfg5d1 = fuzz.gaussmf(dist1, 78.5, 36.7)

# ***********************GAUSSIAN MF'S - Distribution 2*****************************
mfg1d2 = fuzz.gaussmf(dist2, 0.5, 0.083)
mfg2d2 = fuzz.gaussmf(dist2, 1.85, 0.6)
mfg3d2 = fuzz.gaussmf(dist2, 4.2, 1.08)
mfg4d2 = fuzz.gaussmf(dist2, 7.5, 2.08)

#ax2.plot(dist1, mfg1d1, 'r', linewidth=1)
#ax2.plot(dist1, mfg2d1, 'r', linewidth=1)
#ax2.plot(dist1, mfg3d1, 'r', linewidth=1)
ax2.plot(dist1, mfg4d1, 'r', linewidth=1)
ax2.plot(dist1, mfg5d1, 'r', linewidth=1)

ax3.plot(dist2, mfg1d2, 'r', linewidth=1)
ax3.plot(dist2, mfg2d2, 'r', linewidth=1)
ax3.plot(dist2, mfg3d2, 'r', linewidth=1)
ax3.plot(dist2, mfg4d2, 'r', linewidth=1)
"""
# Area of Uncertainity - Dist1
a1gUd1 = mfAU.gaussmfAreaU(dist1, [14, 65.3], [33, 16.3])
a2gUd1 = mfAU.gaussmfAreaU(dist1, [33, 16.3], [44, 21.3])
a3gUd1 = mfAU.gaussmfAreaU(dist1, [44, 21.3], [57.5, 24.1])
a4gUd1 = mfAU.gaussmfAreaU(dist1, [57.5, 24.1], [78.5, 36.7])
"""
# Area of Uncertainity - Dist2
a1gUd2 = mfAU.gaussmfAreaU(dist2, [0.5, 0.083], [1.85, 0.6])
a2gUd2 = mfAU.gaussmfAreaU(dist2, [1.85, 0.6], [4.2, 1.08])
a3gUd2 = mfAU.gaussmfAreaU(dist2, [4.2, 1.08], [7.5, 2.08])
"""
print("\n****************UNCERTAINITY - Distribution 1 - GAUSSIAN MF's**************************")
print("AreaU 1 = ")
print(a1gUd1)
print("AreaU 2 = ")
print(a2gUd1)
print("AreaU 3 = ")
print(a3gUd1)
print("AreaU 4 = ")
print(a4gUd1)
print("Total Area of Uncertainity =")
totAreagUd1 = a1gUd1 + a2gUd1 + a3gUd1 + a4gUd1
print(totAreagUd1)
"""
print("\n****************UNCERTAINITY - Distribution 2 - GAUSSIAN MF's**************************")
print("AreaU 1 = ")
print(a1gUd2)
print("AreaU 2 = ")
print(a2gUd2)
print("AreaU 3 = ")
print(a3gUd2)
print("Total Area of Uncertainity =")
totAreagUd2 = a1gUd2 + a2gUd2 + a3gUd2
print(totAreagUd2)
"""
# Area of Certainity - Dist1
a1gCd1 = mfAC.gaussmfAreaC(dist1, [14, 65.3], [33, 16.3], 0, a1gUd1)
a2gCd1 = mfAC.gaussmfAreaC(dist1, [33, 16.3], [44, 21.3], a1gUd1, a2gUd1)
a3gCd1 = mfAC.gaussmfAreaC(dist1, [44, 21.3], [57.5, 24.1], a2gUd1, a3gUd1)
a4gCd1 = mfAC.gaussmfAreaC(dist1, [57.5, 24.1], [78.5, 36.7], a3gUd1, a4gUd1)
a5gCd1 = mfAC.gaussmfAreaC(dist1, [78.5, 36.7], [0, 0], a4gUd1, 0)
"""
# Area of Certainity - Dist2
a1gCd2 = mfAC.gaussmfAreaC(dist2, [0.5, 0.083], [1.85, 0.6], 0, a1gUd2)
a2gCd2 = mfAC.gaussmfAreaC(dist2, [1.85, 0.6], [4.2, 1.08], a1gUd2, a2gUd2)
a3gCd2 = mfAC.gaussmfAreaC(dist2, [4.2, 1.08], [7.5, 2.08], a2gUd2, a3gUd2)
a4gCd2 = mfAC.gaussmfAreaC(dist2, [7.5, 2.08], [0, 0], a3gUd2, 0)
"""
print("******************CERTAINITY - Distribution 1 - GAUSSIAN MF's**************************")
print("AreaU 1 = ")
print(a1gCd1)
print("AreaU 2 = ")
print(a2gCd1)
print("AreaU 3 = ")
print(a3gCd1)
print("AreaU 4 = ")
print(a4gCd1)
print("AreaU 5 = ")
print(a5gCd1)
print("Total Area of Certainity =")
totAreagCd1 = a1gCd1 + a2gCd1 + a3gCd1 + a4gCd1 + a5gCd1
print(totAreagCd1)
"""
print("******************CERTAINITY - Distribution 2 - GAUSSIAN MF's**************************")
print("AreaU 1 = ")
print(a1gCd2)
print("AreaU 2 = ")
print(a2gCd2)
print("AreaU 3 = ")
print(a3gCd2)
print("AreaU 4 = ")
print(a4gCd2)
print("Total Area of Certainity =")
totAreagCd2 = a1gCd2 + a2gCd2 + a3gCd2 + a4gCd2
print(totAreagCd2)
"""
print("\nfactor for GAUSSIAN MEMBERSHIP FUNCTION- Distribution 1")
print(totAreagCd1/totAreagUd1)
"""
print("\nfactor for GAUSSIAN MEMBERSHIP FUNCTION- Distribution 2")
print(totAreagCd2/totAreagUd2)

# Show plot
#plt.tight_layout()

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
print(len(dist2))
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
a4tpCd1 = mfAC.trapmfAreaC(dist1, [44.0, 50.7, 64.1, 71.0], [63.0, 70.7, 83.1, 94.0], a3tpUd1, a4tpUd1)
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

plt.show()
