import numpy as np
import sympy as sp
import skfuzzy as fuzz
from mfParameters import*
from scipy.integrate import quad
from math import cos, exp, pi

# Using sympy
param = parameters([[0, 1],[.4,3]])
gausParam = param.gaussParam()
print(gausParam)

print("\n\nAREA TEST using SYMPY")


def f1(x):
    return sp.exp(-((x - gausParam[1][0])**2.) / (2 * gausParam[1][1]**2.))
def f2(x):
    return sp.exp(-((x - gausParam[0][0])**2.) / (2 * gausParam[0][1]**2.))

x = sp.Symbol('x')
print(sp.integrate(f1(x), (x, 0, .9501)))
print(sp.integrate(f2(x), (x,.9501, 10)))
print("Total area : {}".format(-1*sp.integrate(f1(x), (x, 0, .9501))+ sp.integrate(f2(x), (x,.9501, 10))))

#------------------------------------Point of intersection 1------------------------------------------------

paramtest1 = parameters([[0, 1],[.4, 3]])
gausParamtest1 = paramtest1.gaussParam()
print(gausParamtest1)
function1= lambda x: exp(-((x - gausParamtest1[0][0]**2.) / (2 * gausParamtest1[0][1]**2.)))
function2= lambda x: exp(-((x - gausParamtest1[1][0]**2.) / (2 * gausParamtest1[1][1]**2.)))
print("\n\nAREA TEST using QUAD INTEGRATE IN SCIPY")
result1, error1 = quad(function1, .950132, 10)
result2, error2 = quad(function2, 0, .950132)
print("Error1 : {} , Result1 : {}".format(error1, result1))
print("Error2 : {} , Result2 : {}".format(error2, result2))
print("Total Error : {} , Total Result : {}".format(error2+error1, result2+result1))

#------------------------------------Point of intersection 2------------------------------------------------
