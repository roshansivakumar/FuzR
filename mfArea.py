"""
Class that contains functions to obtain both the cetainity and uncertainity reigon
"""
import numpy as np
import warnings
from scipy.stats import norm
import scipy.integrate as integrate
from scipy.integrate import quad


class mfAreaUncertainity:

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

    def gaussmfAreaU(x, params1, params2):
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
                print("All Points of intersection")
                print(result)
                if result[i] < 0:
                    result[i] = 10000
                    flag = 0
                elif np.min(result)<i:
                    flag = 1
            if flag == 0:
                r = np.min(result)
            elif flag == 1:
                r = np.max(result)
            else:
                r = np.min(result)

            print("POINT OF INTERSECTION")
            print(r)
            """
            Using CDF
            area = norm.cdf(10, m1, s1) - norm.cdf(r, m1, s1) + norm.cdf(r, m2, s2) - norm.cdf(0, m2, s2)
            """
            print("m2: {} , s2: {} , m1: {} , s1: {}".format(m2, s2, m1, s1))
            function1= lambda x: np.exp(-((x - m2**2.) / (2 * s2**2.)))
            function2= lambda x: np.exp(-((x - m1**2.) / (2 * s1**2.)))
            print("AREA INTEGRATE IN SCIPY")
            result1, error1 = quad(function1, 0, r)
            result2, error2 = quad(function2, r, 10)
            print("Theta 1 - Result1 : {}".format(result1))
            print("Theta 2 - Result2 : {}".format(result2))
            print("Total Result : {} \n--------------------------".format(result2+result1))
            area = result2 + result1
        return area

    def trapmfAreaU(x, params1, params2):
        assert len(params1) == 4, 'a1,b1,c1,d1 parameter must have exactly four\
                                   elements'
        assert len(params2) == 4, 'a2,b2,c2,d2 parameter must have exactly four\
                                   elements'
        a1, b1, c1, d1 = np.r_[params1]
        a2, b2, c2, d2 = np.r_[params2]

        py = (a2 - d1)/(c1 - d1 - b2 + a2)
        # Area
        area = py*(d1 - a2)/2
        return area


class mfAreaCertainity:

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

        #print("Check previous and next Uncertainity")
        #print(pUnA)
        #print(nUnA)
        #print("Check cdf Values")
        #print(norm.cdf(10, m1, s1))
        #print(norm.cdf(0, m1, s1))
        #print("Check mean and standard deviation")
        #print(m1)
        #print(s1)
        area = norm.cdf(10, m1, s1) - norm.cdf(0, m1, s1) - pUnA - nUnA
        #area = 1 - pUnA - nUnA
        return area

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
