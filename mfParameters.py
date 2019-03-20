import numpy as np


class parameters:
    """
    This class contains functions to calculate the parameters of different membership
    functions if given intervals
    """
    global param

    def __init__(self, intervals):
        self.intervals = intervals

    def checkChange(self):
        """
        Checking if the range of the distribution os greater than 0-10
        If it is it will normalize it to 0-10
        """
        intervalsMod = []
        flag = 0
        l = len(self.intervals)
        for i in range(l):
            for j in range(2):
                if (self.intervals[i][j] > 10 or self.intervals[i][j] < 0):
                    intervalsMod = self._minmaxNorm()
                    flag = 1
                    break
        if(flag == 0):
            return self.intervals
        else:
            return intervalsMod


    def _minmaxNorm(self):
        """
        min max normalization to bring intervals into 0-10 range.
        """
        l = len(self.intervals)
        min = 0
        max = 0
        min = np.min(np.min(self.intervals))
        max = np.max(np.max(self.intervals))
        for i in range(l):
            for j in range(2):
                self.intervals[i][j] = ((self.intervals[i][j] - min)/(max-min))*10
        return self.intervals


    def triParam(self):
        """
        Function that accepts the intervals declared by the user and returns the
        parameters - a, b, c for the triangular mrmbership function
        """
        param = []
        l = len(self.intervals)
        for i in range(l):
            a = self.intervals[i][0]
            b = (self.intervals[i][0] + self.intervals[i][1])/2
            c = self.intervals[i][1]
            param.append([a, b, c])
        return param

    def trapParam(self):
        """
        Function that accepts the intervals declared by the user and returns the
        parameters - a, b, c, d for the trapezoidal membership function
        """
        param = []
        l = len(self.intervals)
        for i in range(l):
            a = self.intervals[i][0]
            b = self.intervals[i][0] + 0.25*(self.intervals[i][1] - self.intervals[i][0])
            c = self.intervals[i][1] - 0.25*(self.intervals[i][1] - self.intervals[i][0])
            d = self.intervals[i][1]
            param.append([a, b, c, d])
        return param

    def gaussParam(self):
        """
        Function that accepts the intervals declared by the user and returns the
        parameters - mean and sigma for the Gaussian membership function
        """
        param = []
        l = len(self.intervals)
        for i in range(l):
            m = (self.intervals[i][0] + self.intervals[i][1])/2
            sd = (self.intervals[i][1] - self.intervals[i][0])/4
            param.append([m, sd])
        return param
