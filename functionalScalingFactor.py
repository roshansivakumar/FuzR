import numpy as np
import statistics as stat

x1 = 19.63
y1 = 14.166
z1 = 7.029

x2 = 13.72
y2 = 9.766
z2 = 6.376

sample1 = (x1, y1, z1)
sample2 = (x2, y2, z2)

# statistical measures
print("The Standard Deviation of Sample 1 is % s" %(stat.stdev(sample1)))
print("The Standard Deviation of Sample 2 is % s" %(stat.stdev(sample2)))
print("Variance of the sample 1 is % s" %(stat.variance(sample1)))
print("Variance of the sample 2 is % s" %(stat.variance(sample2)))
print("pvariance of the sample 1 is % s" %(stat.pstdev(sample1)))
print("pvariance of the sample 2 is % s" %(stat.pstdev(sample2)))
print("harmonic mean of the sample 1 is % s" %(stat.harmonic_mean(sample1)))
print("harmonic mean of the sample 2 is % s" %(stat.harmonic_mean(sample2))) 
