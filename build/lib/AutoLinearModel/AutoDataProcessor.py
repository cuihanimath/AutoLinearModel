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

        self._checkAssumptions()


    def _checkAssumptions(self,):
        checkerAssumptionOfDegreesOfFreedom = AssumptionOfDegreesOfFreedom(self.n, self.k,)
        checkerAssumptionOfLinearRelationship = AssumptionOfLinearRelationship(self.correlationMatrix, self.n, self.variableNameList,)
        checkerAssumptionOfNoCollinearity = AssumptionOfNoCollinearity(self.correlationMatrix, self.variableNameList,)
        checkerAssumptionOfNormallyDistributedResiduals = AssumptionOfNormallyDistributedResiduals(self.residuals, self.n, )
        checkerAssumptionOfZeroMeanOfResiduals = AssumptionOfZeroMeanOfResiduals(self.residuals, self.n, )
        checkerAssumptionOfIndependentResiduals = AssumptionOfIndependentResiduals(self.residuals, self.n, )
        checkerAssumptionOfhomoscedasticity = AssumptionOfhomoscedasticity(self.residuals, self.n, )

        self.assumptionViolationDict = {
            0: checkerAssumptionOfDegreesOfFreedom,
            1: checkerAssumptionOfLinearRelationship,
            2: checkerAssumptionOfNoCollinearity,
            3: checkerAssumptionOfNormallyDistributedResiduals,
            4: checkerAssumptionOfZeroMeanOfResiduals,
            5: checkerAssumptionOfIndependentResiduals,
            6: checkerAssumptionOfhomoscedasticity}
        pass

    def checkAssumptions(self):
        for i in range(7):
            print('='*90)
            print (' '*35+'Assumption %s'%(i+1))
            self.assumptionViolationDict[i].log()


    def autoTransformation(self):
        dfX = pd.DataFrame(self.X, columns=self.variableNameList)

        violationList = [self.assumptionViolationDict[key].violation for key in self.assumptionViolationDict.keys()]

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
        return self.dfX.values


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

    from utility import generateRandomData

    X, y = generateRandomData(n=100)

    print('='*90)
    AutoDataProcessorObj = AutoDataProcessor(X, y)

    print('='*90)
    AutoDataProcessorObj.checkAssumptions()

    print('='*90)
    print('Start Transform data')
    AutoDataProcessorObj.autoTransformation()

    #print('='*90)
    #AutoDataProcessor(AutoDataProcessorObj.dfX, y, variableNameList=AutoDataProcessorObj.dfX.columns)
