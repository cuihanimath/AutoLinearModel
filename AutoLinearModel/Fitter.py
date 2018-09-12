import numpy as np
import tensorflow as tf
import scipy
from AutoLinearModel.utility import AutoSummarizer
from IPython.display import display
import pandas as pd


class Fitter(object):
    def __init__(self, fit_method='matrix', n_features=None, **kwargs):
        self.fit_method = fit_method
        self.params = None

        self.n_features = n_features

        if self.fit_method == "gradient":
            self.tf_lr = tf_linear_regression()

    def fit(self, x, y, return_params=True, **kwargs):
        self.data = {'x': x, 'y': y}
        self.n_features = x.shape[1]

        if self.fit_method == "matrix":
            print("Utilizing matrix to fit data")
            self.matrixFit(x=x, y=y, **kwargs)
        elif self.fit_method == 'gradient':
            print("Utilizing gradient descent to fit data")
            self.tf_lr.fit(x=x, y=y, **kwargs)
            self.params = self.tf_lr.params

        if return_params:
            return self.params

    def predict(self, x):
        if self.fit_method == 'gradient':
            return self.tf_lr.predict(x)
        elif self.fit_method == 'matrix':
            Weight = self.params['Weight']
            Bias = self.params['Bias']
            return x @ Weight + Bias

    def summary(self):
        x = self.data['x']
        y_pred = self.predict(x)
        y = self.data['y']
        residuals = y - y_pred
        n_features = self.n_features

        Summarizer = AutoSummarizer()

        SSE = Summarizer.calculateSSE(residuals)
        AIC = Summarizer.calculateAIC(residuals, n_features)
        BIC = Summarizer.calculateBIC(residuals, n_features)

        R_squared = Summarizer.calculateRsquared(y, y_pred, n_features, adjusted=False)
        R_squared_adjusted = Summarizer.calculateRsquared(y, y_pred, n_features, adjusted=True)
        # LogLikelihood = Summarizer.calculateLogLikelihood(y, y_pred)

        F_stat = Summarizer.calculateF(y, y_pred, n_features)
        # T_stat = Summarizer.calculateT(x, residuals, n_features, self.params)
        Skewness = Summarizer.calculateSkewness(y, y_pred)
        # Omnibus = Summarizer.calculateOmnibus(Skewness['Skewness'], Skewness['Kurtosis'], n=len(x))

        DurbinWatson = Summarizer.calculateDurbinWatson(residuals)
        JB_stat = Summarizer.calculateJB(Skewness['Skewness'], Skewness['Kurtosis'], n=len(x))
        ConditionNum = Summarizer.calculateConditionNumber(x)

        # print("Dur", DurbinWatson)

        df_metric = pd.DataFrame({"SSE": SSE, "AIC": AIC, "BIC": BIC, "R_squared": R_squared, 
            "R_squared_adjusted": R_squared_adjusted, "DurbinWatson": DurbinWatson,
            "F_stat": F_stat['F_value'],"F_stat P_value":F_stat['P_value'],"" "ConditionNum": ConditionNum
            }, index=["value"])

        def get_df_CI_coef(CI_coeff):

            value = CI_coeff['Weight']
            value.insert(0, CI_coeff['Bias'])
            n_beta = len(value)
            col = ["beta_" + str(i) for i in range(n_beta)]
            df_CI_coef = pd.DataFrame(value).T
            df_CI_coef.columns = col
            df_CI_coef.index = ['lower_bound', 'upper_bound']
            return df_CI_coef

        df_CI_coef = get_df_CI_coef(self.CI_coefficient())

        display("Regression summary with {} fit_method".format(self.fit_method))
        display(df_metric)
        display(df_CI_coef)

    def lambda_add_1s(self, x): return np.array([np.append(arr, 1) for arr in x])

    def matrixFit(self, x, y):
        """
        Function: use matrix manipulation to calculate the parameters
        Input:
            x: numpy array, dimension should be (n,m)
            y: numpy array, dimension should be (n,)
        Output:
            parameters of Weights
        """
        tmp_x = self.lambda_add_1s(x)
        self.params = np.linalg.inv(np.transpose(tmp_x)@tmp_x)@np.transpose(tmp_x)@y
        self.params = {"Weight": self.params[:-1], 'Bias': self.params[-1]}

    def estimate_error_std(self):
        x = self.data['x']
        y = self.data['y']
        y_pred = self.predict(x)
        self.error_std = np.std(y_pred - y)
        return self.error_std

    def CI_prediction(self, confidence=0.95):
        if not self.n_features:
            n_features = 1
        else:
            n_features = self.n_features

        error_std = self.estimate_error_std()
        y = self.data['y']

        t_value = scipy.stats.t.ppf(1 - (1 - confidence) / 2, df=len(y) - n_features - 1)

        self.CI_pred = list(map(lambda x: [x - t_value * error_std, x + t_value * error_std], y))
        return self.CI_pred

    def CI_coefficient(self, confidence=0.95):

        x = self.lambda_add_1s(self.data['x'])
        y = self.data['y']

        if not self.n_features:
            n_features = 1
        else:
            n_features = self.n_features

        error_std = self.estimate_error_std()

        cov_matrix = error_std**2 * np.linalg.inv(np.transpose(x) @ x)


        t_value = scipy.stats.t.ppf(1 - (1 - confidence) / 2, df=len(x) - n_features - 1)

        if self.fit_method == "gradient":
            Weight = np.append(self.params['Weight'].flatten(), self.params['Bias'])
        else:
            Weight = np.append(self.params['Weight'], self.params['Bias'])

        tmp_CI = []

        for i in range(n_features + 1):
            try:   
                std = np.sqrt(cov_matrix[i, i])
            except:
                raise Exception("data is not positive define")

            tmp_CI.append([Weight[i] - t_value * std, Weight[i] + t_value * std])

        self.CI_coef = {"Weight": tmp_CI[:-1], "Bias": tmp_CI[-1]}

        return self.CI_coef


class tf_linear_regression(object):

    def fit(self, x, y, learning_rate=0.01, epochs=10, display_freq=10):

        if len(x.shape) == 1:
            self.n_features = 1
        else:
            self.n_features = x.shape[1]

        self.learning_rate = 0.01
        self.params = {}
        self.sess = None

        self.X = tf.placeholder("float", shape=[None, self.n_features])
        self.Y = tf.placeholder("float")

        self.learning_rate = tf.placeholder("float")

        self.W = tf.Variable(np.random.randn(self.n_features, 1), name="Weight")
        self.b = tf.Variable(np.random.randn(), name="Bias")
        self.W = tf.cast(self.W, tf.float32)

        self.pred = tf.matmul(self.X, self.W) + self.b

        self.pred = tf.squeeze(self.pred)
        self.cost = tf.reduce_mean(tf.pow(self.Y - self.pred, 2))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()


        if x.shape[1] != self.n_features:
            raise Exception("n_features is not the same as input x features")
        if self.n_features == 1:
            x = x.reshape(-1, 1)
        if self.sess:
            pass
        else:
            self.sess = tf.Session()
            self.sess.run(self.init)

        dic_input = {self.X: x, self.Y: y, self.learning_rate: learning_rate}

        for epoch in range(epochs):

            self.sess.run(self.optimizer, feed_dict=dic_input)

            if epoch % display_freq == 0:
                _cost = self.sess.run(self.cost, feed_dict=dic_input)
                print("epoch:{}, cost:{}".format(epoch, _cost))

        self.params["Weight"] = self.sess.run(self.W)
        self.params["Bias"] = self.sess.run(self.b)

    def partial_fit(self,x,y):

        if x.shape[1] != self.n_features:
            raise Exception("n_features is not the same as input x features")
        if self.n_features == 1:
            x = x.reshape(-1, 1)
        if self.sess:
            pass
        else:
            self.sess = tf.Session()
            self.sess.run(self.init)

        dic_input = {self.X: x, self.Y: y, self.learning_rate: learning_rate}

        for epoch in range(epochs):

            self.sess.run(self.optimizer, feed_dict=dic_input)

            if epoch % display_freq == 0:
                _cost = self.sess.run(self.cost, feed_dict=dic_input)
                print("epoch:{}, cost:{}".format(epoch, _cost))

        self.params["Weight"] = self.sess.run(self.W)
        self.params["Bias"] = self.sess.run(self.b)

    def predict(self, x):

        y_pred = self.sess.run(self.pred, feed_dict={self.X: x})
        return y_pred
