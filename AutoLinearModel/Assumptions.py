from __future__ import print_function
from builtins import super

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from HypothesisTest import HypothesisTest

class Assumption(object):
	def __init__(self):
		self.name = None
		self.violation = False
		self.remedies = None
			
	def check(self):
		pass
	
	def log(self):
		print('-'*90)
		if self.violation == True:
			print('Assumption: ' + self.name + ', is violated.')
			print('Recommended remedies: ' + self.remedies)
		else:
			print('Assumption: ' + self.name + ', is not violated.')

class AssumptionOfDegreesOfFreedom(Assumption):
	def __init__(self, n, k, printLot=True):
		super().__init__()
		self.name = 'The degrees of freedom should be larger than 0'
		self.remedies = 'Add more data or reduce the dimensionality of the feature space.'
		self.check(n, k)
		
	def check(self, n, k):
		degreesOfFreedom = n - k - 1
		if degreesOfFreedom > 0:
			self.violation = False
		else:
			self.violation = True
		
class AssumptionOfLinearRelationship(Assumption):
	def __init__(self, correlationMatrix, n, variableNameList):
		super().__init__()
		self.name = 'The relationship between the independent and dependent variables to be linear'
		self.remedies = 'Transform or remove independent variables'
		self.check(correlationMatrix, n)
		self.variableNameList = np.array(variableNameList)
		
	def check(self, correlationMatrix, n):
		correlationList = correlationMatrix[:-1,-1]
		calculateTStats = lambda r: r*np.sqrt(n-2)/float(np.sqrt(1-r**2))
		calculatePValue = lambda tStat: stats.t.sf(np.abs(tStat), n-2)*2 

		tStatList = map(calculateTStats, correlationList)
		pValueList = list(map(calculatePValue, tStatList))
		alpha = 0.05

		self.idxOfCorrelatedFeatures = np.where((np.array(pValueList) <= alpha) == True)[0]
		self.idxOfUncorrelatedFeatures = np.where((np.array(pValueList) > alpha) == True)[0]
		if len(self.idxOfUncorrelatedFeatures) != 0:
			self.violation = True
		else:
			self.violation = False
			
	def log(self):
		print('-'*90)
		if self.violation == True:
			print('Assumption: ' + self.name + ', is violated.')
			print('Recommended remedies: ' + self.remedies + 'Of features:')
			print(self.variableNameList[self.idxOfUncorrelatedFeatures])
		else:
			print('Assumption: ' + self.name + ', is not violated.')
		
class AssumptionOfNoCollinearity(Assumption):
	def __init__(self, correlationMatrix, variableNameList):
		super().__init__()
		self.name = 'There should not be collinearity among independent variables'
		self.remedies = 'Transform or remove independent variables'
		self.check(correlationMatrix)
		self.variableNameList = np.array(variableNameList)

	def check(self, correlationMatrix):
		VIF = np.linalg.inv(correlationMatrix[:-1, :-1])
		self.vifList = VIF.diagonal()
		
		maxVIF = max(self.vifList)
		if maxVIF > 1:
			self.violation = True
		else:
			self.violation = False
			
	def log(self):
		print('-'*90)
		if self.violation == True:
			print('Assumption: ' + self.name + ', is violated.')
			print('Recommended remedies: ' + self.remedies + 'Of features whose VIF > 1.')
			print("""		VIF = 1 (Not correlated);\n		1 < VIF < 5 (Moderately correlated);\n		VIF >=5 (Highly correlated)""")
			print("""vifList:""")
			print(self.vifList)
			
		else:
			print('Assumption: ' + self.name + ', is not violated.')


class AssumptionOfNormallyDistributedResiduals(Assumption):
	def __init__(self, residuals, n):
		super().__init__()
		self.name = 'The residuals of the model should be normally distributed'
		self.remedies = 'Transform some features, because the linearity assumption may be violated or the distributions of some of the variables that are random are extremely asymmetric or long-tailed.'
		self.check(residuals, n)
		
	def check(self, residuals, n):
		if n < 200:
			residuals = np.random.normal(size=100)
			fig = sm.qqplot(residuals, stats.norm, fit=True, line='45')
			plt.title('QQ plot')
			plt.show()
		
		WStats, pValue = shapiro(residuals)
		HypothesisTestObj = HypothesisTest(H0="""the population is normally distributed""", pValue=pValue)
		#HypothesisTestObj.log()
		
		self.violation = 1-HypothesisTestObj.result
		

class AssumptionOfZeroMeanOfResiduals(Assumption):
	def __init__(self, residuals, n):
		super().__init__()
		self.name = 'The mean of the residuals should be zero'
		self.remedies = 'Transform some features, because the linearity assumption may be violated or the distributions of some of the variables that are random are extremely asymmetric or long-tailed.'
		self.check(residuals, n)
		
	def check(self, residuals, n):
		tStats = (np.mean(residuals))/(np.std(residuals)/np.sqrt(n))
		pValue = stats.t.sf(np.abs(tStats), n-1)*2
		
		HypothesisTestObj = HypothesisTest(H0="""the mean of normally distributed residuals is 0""", pValue=pValue)
		#HypothesisTestObj.log()
		
		self.violation = 1-HypothesisTestObj.result
		
class AssumptionOfIndependentResiduals(Assumption):
	def __init__(self, residuals, n):
		super().__init__()
		self.name = 'The residuals of the model should be independent'
		self.remedies = 'Use a time series model rather than a linear regression model to model the data'
		self.check(residuals, n)
		
	def check(self, residuals, n):
		if n < 200:
			plot_acf(residuals)
			plt.title('Autocorrelation Plot')
			plt.show()
		
		print('Implement Durbin Watson to checke the autocorrelation.')
		tStats = durbin_watson(residuals)
		pValue = stats.t.sf(np.abs(tStats), n-1)*2

		HypothesisTestObj = HypothesisTest(H0="""the residuals are not correlated""", pValue=pValue)
		#HypothesisTestObj.log()
		
		self.violation = 1-HypothesisTestObj.result
		
class AssumptionOfhomoscedasticity(Assumption):
	def __init__(self, residuals, n):
		super().__init__()
		self.name = 'The residuals should have constant variance'
		self.remedies = 'Transform the target variable.'
		self.check(residuals, n)
		
	def check(self, residuals, n):
		print('-'*90)
		if n < 200:
			plt.plot(residuals)
			plt.title('residuals histogram')
			plt.show()
			
		print('Use Levene test because it is less sensitive than the Bartlett test to departures from normality.')
	   
		#split the data into chunks of size 40
		chunks = []
		size = 50
		for i in range(0, n, size):
			temp = residuals[i:i+size].reshape(-1,)
			if len(temp) == size:
				chunks.append(temp)
			else:
				chunks[-1] = residuals[i-size:]

		print('generate %s chunks for Levene test'%len(chunks))
		statistics, pValue = stats.levene(*chunks)
		
		HypothesisTestObj = HypothesisTest(H0='Variance of each subsets of data is the same', pValue=pValue)
		#HypothesisTestObj.log()
		
		self.violation = 1-HypothesisTestObj.result



if __name__ == '__main__':
	from sklearn.linear_model import LinearRegression
	
	n = 500
	x0 = (np.random.random(size=n)*2-1).reshape(-1,1)
	x1 = (np.random.random(size=n)*2-1).reshape(-1,1)
	x2 = (np.random.random(size=n)*2-1).reshape(-1,1)
	X = np.concatenate([x0, x1, x2], axis=1)
	y = -x0 + x1**2 + x2
	
	k = X.shape[1]
	variableNameList = ['x0', 'x1', 'x2']
	correlationMatrix = np.corrcoef(X.T)
	
	regr = LinearRegression()
	regr.fit(X, y)
	residuals = y - regr.predict(X)
	
	AssumptionOfDegreesOfFreedomObj = AssumptionOfDegreesOfFreedom(n=100, k=10)
	AssumptionOfLinearRelationshipObj = AssumptionOfLinearRelationship(correlationMatrix, n, variableNameList)
	AssumptionOfNoCollinearityObj = AssumptionOfNoCollinearity(correlationMatrix, variableNameList,)
	AssumptionOfNormallyDistributedResidualsObj = AssumptionOfNormallyDistributedResiduals(residuals, n)
	AssumptionOfZeroMeanOfResidualsObj = AssumptionOfZeroMeanOfResiduals(residuals, n)
	AssumptionOfIndependentResidualsObj = AssumptionOfIndependentResiduals(residuals, n)
	AssumptionOfhomoscedasticityObj = AssumptionOfhomoscedasticity(residuals, n)
	
	AssumptionOfDegreesOfFreedomObj.log()
	AssumptionOfLinearRelationshipObj.log()
	AssumptionOfNoCollinearityObj.log()
	AssumptionOfNormallyDistributedResidualsObj.log()
	AssumptionOfZeroMeanOfResidualsObj.log()
	AssumptionOfIndependentResidualsObj.log()
	AssumptionOfhomoscedasticityObj.log()