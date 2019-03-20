"""
Class that contains functions to obtain both the cetainty and uncertainty reigon

NOTE
- For the concept of FuzR using CDF or PDF to calculate the Areas will not work as the
CDF and PDF are basically the integration of the probability density function which is
a curve that represents the probability and not absolute value, between two limits.
Hence in our case we need to integrate between points to get the area under the curve

TODO
1. Robust algorithm for calculating Uncertainty and Certainty area of Gaussian membership
functions using Integration.

"""
import numpy as np
import warnings
from scipy.stats import norm
import sympy
from sympy.core.function import nfloat
from scipy.optimize import fsolve


class mfAreaUncertainity:
    """
    class that contains methods to calculate the area of unceratinty
    of similar and adjacent triangualr, trapezoidal and gaussian membership functions
    """

    def trimfAreaU(x, params1, params2):

        assert len(params1) == 3, 'abc parameter must have exactly three elements.'
        assert len(params2) == 3, 'fgh parameter must have exactly three elements.'
        a, b, c = np.r_[params1]     # Zero-indexing in Python
        f, g, h = np.r_[params2]     # Zero-indexing in Python
        assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'
        assert f <= g and g <= h, 'fgh requires the three elements a <= b <= c.'
        # Point of Intersection
        x1 = b
        y1 = 1
        x2 = c
        y2 = 0
        x3 = f
        y3 = 0
        x4 = g
        y4 = 1

        try:  # find parameters that make denominator zero
            px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
            py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        except RuntimeWarning:  # Used since, I was getting / 0 runtimeWarning
            print(a, b, c, f, g, h)
        pt = [px, py]

        # Area
        area = 0.5*py*(c-f)
        """
        Alternative - using Clamers rule
        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x,y
        L1 = line([4,1], [6,0])
        L2 = line([5,0], [8,1])
        R = intersection(L1, L2)
        """

        return area

    def gaussmfAreaU(x, params1, params2, prevInter):
        assert len(params1) == 2, 'sigma,c parameter must have exactly two \
                                   elements.'
        assert len(params2) == 2, 'sigma,c parameter must have exactly two \
                                   elements.'
        m1, s1 = np.r_[params1]     # Zero-indexing in Python
        m2, s2 = np.r_[params2]     # Zero-indexing in Python

        # Solving coeffiecients of quadratic poly
        def solve(m1, m2, std1, std2):
            a = 1/(2*std1**2) - 1/(2*std2**2)
            b = m2/(std2**2) - m1/(std1**2)
            c = m1**2 / (2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
            return np.roots([a, b, c])

        flag = 2
        result = solve(m1, m2, s1, s2)
        if len(result) == 0:  # Completely non-overlapping
            area = 0.0
            return area
        if len(result) > 0:  # One point of contact
            for i in range(len(result)):
                #print("All Points of intersection")
                #print(result)
                if result[i] < 0:
                    result[i] = 10000
                    flag = 0
            if np.min(result) > prevInter :
                r = np.min(result)
            else:
                r = np.max(result)
            if flag == 0:
                r = np.min(result)

            #print("POINT OF INTERSECTION")
            #print(r)

        x = sympy.var("x")
        func1 = sympy.exp(-(((x-params1[0])/params1[1])**2)/2)
        func2 = sympy.exp(-(((x-params2[0])/params2[1])**2)/2)
        int1 = sympy.integrate(func1, (x, r, 10))
        int2 = sympy.integrate(func2, (x, 0, r))
        area1 = nfloat(int1)
        area2 = nfloat(int2)
        areaTot = round(area1, 3) + round(area2, 3)

        return round(areaTot, 3), r

    def trapmfAreaU(x, params1, params2):
        assert len(params1) == 4, 'a1,b1,c1,d1 parameter must have exactly four\
                                   elements'
        assert len(params2) == 4, 'a2,b2,c2,d2 parameter must have exactly four\
                                   elements'
        a1, b1, c1, d1 = np.r_[params1]
        a2, b2, c2, d2 = np.r_[params2]

        if(b2 < c1):
            area = (c1 - b2 + d1 - a2)/2
        elif(b2==c1):
            area = (d1 - a2)/2
        else:
            py = (a2 - d1)/(c1 - d1 - b2 + a2)
            area = py*(d1 - a2)/2

        return area


class mfAreaCertainity:
    """
    class that contains methods to calculate the area of ceratinty
    of triangualr, trapezoidal and gaussian membership functions
    """

    def trimfAreaC(x, params1,pUnA, nUnA):
        assert len(params1) == 3, 'abc parameter must have exactly three \
                                   elements.'
        #assert len(params2) == 3, 'fgh parameter must have exactly three \
        #                           elements.'
        a, b, c = np.r_[params1]     # Zero-indexing in Python
        #f, g, h = np.r_[params2]     # Zero-indexing in Python
        assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'
        #assert f <= g and g <= h, 'fgh requires the three elements a <= b <= c.'
        area = 0.5*(c-a) - pUnA - nUnA
        return area

    def gaussmfAreaC(x, params1, params2, pUnA, nUnA):
        assert len(params1) == 2, 'sigma,c parameter must have exactly two \
                                   elements.'
        assert len(params2) == 2, 'sigma,c parameter must have exactly two \
                                   elements.'
        m1, s1 = np.r_[params1]     # Zero-indexing in Python
        m2, s2 = np.r_[params2]     # Zero-indexing in Python

        x = sympy.var("x")
        func1 = sympy.exp(-(((x-params1[0])/params1[1])**2)/2)
        int1 = sympy.integrate(func1, (x, 0, 10))
        area1 = nfloat(int1)
        areaTot = round(area1, 3) - pUnA - nUnA

        return round(areaTot, 3)

    def trapmfAreaC(x, params1, params2, pUnA, nUnA):
        assert len(params1) == 4, 'a1,b1,c1,d1 parameter must have exactly four\
                                   elements'
        assert len(params2) == 4, 'a2,b2,c2,d2 parameter must have exactly four\
                                   elements'
        a1, b1, c1, d1 = np.r_[params1]
        a2, b2, c2, d2 = np.r_[params2]

        # Area
        totAr = (c1 - b1 + d1 - a1)/2
        area = totAr - pUnA - nUnA
        return area
