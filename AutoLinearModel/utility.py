import math
import numpy as np
import scipy
"""
Plz check http://connor-johnson.com/2014/02/18/linear-regression-with-python/ for the fomula used below
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def calculateBIC(X, y):
    regr = LinearRegression()
    regr.fit(X, y)

    n, p = X.shape
    y_hat = regr.predict(X)
    residuals = y - y_hat
    sse = sum(residuals**2)
    BIC = n * np.log(sse / n) + p * np.log(n)  # the smaller is the better
    return BIC


def powerTransformation(x, order):
    if order == 'e':
        return np.exp(x)
    elif order == 's':
        return np.sin(x)
    elif order == 'c':
        return np.cos(x)

    minValue = min(x)
    if minValue > 0:
        minValue = 0
    else:
        minValue -= 1

    if order != 0:
        return (x - minValue)**order / float(order)

    else:
        return np.log(x - minValue + 1)


def calculateAdjustedR2(X, y):
    regr = LinearRegression()
    regr.fit(X, y)
    y_hat = regr.predict(X)

    n, k = X.shape
    return (1 - r2_score(y, y_hat)) * (n - 1) / float(n - k)
    # return r2_score(y, y_hat)


class AutoSummarizer(object):
    def calculateAIC(self, residuals, p):
        n = len(residuals)
        SSE = self.calculateSSE(residuals)
        AIC = 2 * p + n * np.log(SSE / n)
        return AIC

    def calculateBIC(self, residuals, p):
        n = len(residuals)
        SSE = sum(residuals**2)
        BIC = n * np.log(SSE / n) + p * np.log(n)  # the smaller is the better
        return BIC

    def calculateSSE(self, residuals):
        SSE = sum(residuals**2)
        return SSE

    def calculateRsquared(self, y, y_pred, p, adjusted=True):
        n = len(y)
        y_ave = np.mean(y)
        ESS = np.sum(y_pred - y_ave)
        SSE = self.calculateSSE(y - y_pred)
        TSS = ESS + SSE
        if adjusted:
            R_squared_adjusted = 1 - (SSE / (n - p - 1)) / (TSS / n - 1)
            return R_squared_adjusted
        else:
            R_squared = ESS / TSS
            return R_squared

    def calculateLogLikelihood(self, y, y_pred):
        n = len(y)
        SSE = sum((y - y_pred)**2)
        S2 = SSE / n
        L = (1 / np.sqrt(2 * np.pi * S2)) ** n * np.exp(-SSE / (S2 * 2))
        print(L)
        return np.log(L)

    def calculateF(self, y, y_pred, p):
        n = len(y)

        y_ave = np.mean(y)
        SSM = sum((y_pred - y_ave)**2)

        residuals = y - y_pred
        sigma_squared = np.var(residuals)

        SSE = self.calculateSSE(residuals)

        MSM = SSM / sigma_squared / p
        MSE = SSE / sigma_squared / (n - p - 1)
        F = MSM / MSE
        p_value = scipy.stats.f.cdf(F, p, n - p - 1)

        return {"F_value": F, "P_value:": p_value}

    def calculateT(self, x, residuals, p, params):
        n = len(x)
        df = n - p - 1
        Bias = params['Bias']
        Weight = params['Weight']

        error_std = np.sum(residuals**2) / df
        x = self.lambda_add_1s(x)

        cov_matrix = error_std**2 * np.linalg.inv(x.transpose()@x)

        coefficients = np.append(Weight, Bias)
        T_values = []
        P_values = []
        std_errs = []
        for i in range(len(coefficients)):
            tmp_t = coefficients[i] / np.sqrt(cov_matrix[i, i])
            tmp_p = scipy.stats.t.cdf(tmp_t, df=df)
            tmp_std_err = np.sqrt(cov_matrix[i, i])

            T_values.append(tmp_t)
            P_values.append(tmp_p)
            std_errs.append(tmp_std_err)

        return {"t_values": T_values, "p_values": P_values, "std_err": std_errs}

    def lambda_add_1s(self, x):
        return np.array([np.append(arr, 1) for arr in x])

    def calculateSkewness(self, y, y_pred):
        n = len(y)
        numerator_s = np.sum((y - y_pred)**3) / n
        denominator_s = (((y - y_pred)**2) / n)**(3 / 2)
        Skewness = numerator_s / denominator_s

        numerator_k = np.sum((y - y_pred)**4) / n
        denominator_k = ((np.sum((y - y_pred)**2)) / n)**2

        Kurtosis = numerator_k / denominator_k
        return {"Skewness": Skewness, "Kurtosis": Kurtosis}

    def calculateOmnibus(self, Skewness, Kurtosis, n):

        def Z1(s, n):
            Y = s * np.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2.0)))
            b = 3.0 * (n**2.0 + 27.0 * n - 70) * (n + 1.0) * (n + 3.0)
            b /= (n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0)
            W2 = - 1.0 + np.sqrt(2.0 * (b - 1.0))
            alpha = np.sqrt(2.0 / (W2 - 1.0))
            z = 1.0 / np.sqrt(np.log(np.sqrt(W2)))
            z *= np.log(Y / alpha + np.sqrt((Y / alpha)**2.0 + 1.0))
            return z

        def Z2(k, n):
            E = 3.0 * (n - 1.0) / (n + 1.0)
            v = 24.0 * n * (n - 2.0) * (n - 3.0)
            v /= (n + 1.0)**2.0 * (n + 3.0) * (n + 5.0)
            X = (k - E) / np.sqrt(v)
            b = (6.0 * (n**2.0 - 5.0 * n + 2.0)) / ((n + 7.0) * (n + 9.0))
            b *= np.sqrt((6.0 * (n + 3.0) * (n + 5.0)) / (n * (n - 2.0) * (n - 3.0)))
            A = 6.0 + (8.0 / b) * (2.0 / b + np.sqrt(1.0 + 4.0 / b**2.0))
            z = (1.0 - 2.0 / A) / (1.0 + X * np.sqrt(2.0 / (A - 4.0)))
            z = (1.0 - 2.0 / (9.0 * A)) - z**(1.0 / 3.0)
            z /= np.sqrt(2.0 / (9.0 * A))
            return z

        Omnibus = Z1(Skewness, n)**2.0 + Z2(Kurtosis, n)**2.0
        return Omnibus

    def calculateDurbinWatson(self, residuals):
        DW = np.sum(np.diff(residuals)**2.0) / np.sum(residuals**2)
        return DW

    def calculateJB(self, Skewness, Kurtosis, n):

        JB = (n / 6.0) * (Skewness**2.0 + (1.0 / 4.0) * (Kurtosis - 3.0)**2.0)
        p = 1.0 - scipy.stats.chi2(2).cdf(JB)

        return {"JB_statistic": JB, "p_value": p}

    def calculateConditionNumber(self, x):
        return np.linalg.cond(x)
