# -*- coding: utf-8 -*-

class HypothesisTest:
    def __init__(self, H0=None, pValue=0, alpha=0.05):
        self.H0 = H0
        self.alpha = alpha
        self.pValue = pValue 
        self.getResult()
        
    def getResult(self):
        if self.pValue < self.alpha:
            self.result = False 
        else:
            self.result = True #accept H0 i.e. cannot reject H0
    
    def log(self):
        print("""p-value: how likely is it that we’d get a test statistic as extreme as we did if the null hypothesis were true?""")
        if self.result == True:
            print("""P-value is %s larger than the significance level %s, thus we fail to reject, i.e. conclude, the null hypothesis that %s with %s%% confidence."""%(self.pValue, self.alpha, self.H0, 100*(1-self.alpha)))
        else:
            print("""P-value is %s smaller than the significance level %s, thus we reject the null hypothesis that %s with %s%% confidence."""%(self.pValue, self.alpha, self.H0, 100*(1-self.alpha)))
            
            
if __name__ == '__main__':
    HypothesisTestObj = HypothesisTest(H0='test', pValue=0.1)
    HypothesisTestObj.log()