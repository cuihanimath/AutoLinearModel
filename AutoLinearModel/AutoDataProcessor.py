from __future__ import print_function
from builtins import super

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

from Assumptions import *
from utility import *


class AutoDataProcessor:
    def __init__(self, X, y, variableNameList=[]):
        self.X = X
        self.y = y

        self.n, self.k = X.shape

        if len(variableNameList) < self.k:
            variableNameList = ['x' + str(item) for item in range(self.k)]
        self.variableNameList = np.array(variableNameList)
        print('variableNameList', self.variableNameList)

        if self.n > 10000:
            print('size of the sampe is too large')

        self.correlationMatrix = np.corrcoef(np.concatenate([self.X, self.y], axis=1).T)

        regr = LinearRegression()
        regr.fit(X, y)
        y_pred = regr.predict(X)
        self.residuals = y - y_pred

        self.check()

    def check(self,):
        print('=' * 90)
        print(1)
        checkerAssumptionOfDegreesOfFreedom = AssumptionOfDegreesOfFreedom(self.n, self.k,)
        checkerAssumptionOfDegreesOfFreedom.log()

        print('=' * 90)
        print(2)
        checkerAssumptionOfLinearRelationship = AssumptionOfLinearRelationship(self.correlationMatrix, self.n, self.variableNameList,)
        checkerAssumptionOfLinearRelationship.log()

        print('=' * 90)
        print(3)
        checkerAssumptionOfNoCollinearity = AssumptionOfNoCollinearity(self.correlationMatrix, self.variableNameList,)
        checkerAssumptionOfNoCollinearity.log()

        print('=' * 90)
        print(4)
        checkerAssumptionOfNormallyDistributedResiduals = AssumptionOfNormallyDistributedResiduals(self.residuals, self.n, )
        checkerAssumptionOfNormallyDistributedResiduals.log()

        print('=' * 90)
        print(5)
        checkerAssumptionOfZeroMeanOfResiduals = AssumptionOfZeroMeanOfResiduals(self.residuals, self.n, )
        checkerAssumptionOfZeroMeanOfResiduals.log()

        print('=' * 90)
        print(6)
        checkerAssumptionOfIndependentResiduals = AssumptionOfIndependentResiduals(self.residuals, self.n, )
        checkerAssumptionOfIndependentResiduals.log()

        print('=' * 90)
        print(7)
        checkerAssumptionOfhomoscedasticity = AssumptionOfhomoscedasticity(self.residuals, self.n, )
        checkerAssumptionOfhomoscedasticity.log()

        self.assumptionViolationDict = {
            0: checkerAssumptionOfDegreesOfFreedom,
            1: checkerAssumptionOfLinearRelationship,
            2: checkerAssumptionOfNoCollinearity,
            3: checkerAssumptionOfNormallyDistributedResiduals,
            4: checkerAssumptionOfZeroMeanOfResiduals,
            5: checkerAssumptionOfIndependentResiduals,
            6: checkerAssumptionOfhomoscedasticity}
        pass

    def autoTransformation(self):
        dfX = pd.DataFrame(self.X, columns=self.variableNameList)

        violationList = [self.assumptionViolationDict[key].violation for key in self.assumptionViolationDict.keys()]
        print(violationList)

        if (violationList[0]) + (violationList[5]):
            print('No transformation of data')

        elif violationList[2] != 1:
            print('only collinearity violated')
            pass

        else:
            print('only collinearity violated or other also violated, first add interaction')
            for i in range(len(self.variableNameList)):
                for j in range(i + 1, len(self.variableNameList)):
                    dfX[self.variableNameList[i] + '*' + self.variableNameList[j]] = dfX[self.variableNameList[i]] * dfX[self.variableNameList[j]]
            pass

        if (violationList[1] + violationList[3] + violationList[4] + violationList[6]) > 0:
            print('other also violated, add interaction and power')
            for i in range(len(self.variableNameList)):
                for order in ['e', -3, -2, -1.5, -1, 0, 2, 3]:
                    dfX[self.variableNameList[i] + '_order_' + str(order)] = powerTransformation(dfX[self.variableNameList[i]], order)
            pass

        remaining = list(self.variableNameList)
        maxCorrelatedFeature = remaining[np.argmax(abs(self.correlationMatrix[:-1, -1]))]
        selected = [maxCorrelatedFeature]
        remaining.remove(maxCorrelatedFeature)
        currentScore = 0
        bestNewScore = 0

        while len(remaining) != 0 and (currentScore == bestNewScore):
            scoresList = []
            for candidate in remaining:
                X_temp = dfX[np.append(selected, candidate)]

                score = calculateBIC(X_temp, self.y)
                #score = calculateR2(X_temp, self.y)
                scoresList.append((score, candidate))

            scoresList.sort(reverse=True)
            bestNewScore, bestCandidate = scoresList.pop()

            if bestNewScore < currentScore:
                remaining.remove(bestCandidate)
                selected.append(bestCandidate)
                currentScore = bestNewScore

        self.dfX = dfX[selected]
        print('selected features are:', selected)
        return selected


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    """
	X
	independent variable
	experimental
	predictor variable

	y
	dependent variable
	outcome
	"""

    n = 500

    x0 = (np.random.random(size=n) * 2 - 1).reshape(-1, 1)
    x1 = (np.random.random(size=n) * 2 - 1).reshape(-1, 1)
    x2 = (np.random.random(size=n) * 2 - 1).reshape(-1, 1)
    x3 = (np.random.random(size=n) * 2 - 1).reshape(-1, 1)
    x4 = 3 * x2 - x1
    x5 = np.random.rand(n).reshape(-1, 1)

    X = np.concatenate([x0, x1, x2, x3, x4], axis=1)
    y = -x0 + x1**4 + x2 + 0 * x3 + x4 + x5

    print(X.shape)
    regr = LinearRegression().fit(X, y)
    y_pred = regr.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(mse, r2)

    AutoDataProcessorObj = AutoDataProcessor(X, y)

    AutoDataProcessorObj.autoTransformation()

    regr = LinearRegression().fit(AutoDataProcessorObj.dfX, y)
    y_pred = regr.predict(AutoDataProcessorObj.dfX)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(mse, r2)

    AutoDataProcessor(AutoDataProcessorObj.dfX, y, variableNameList=AutoDataProcessorObj.dfX.columns)
