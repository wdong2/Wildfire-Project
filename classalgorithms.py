from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import pickle
import matplotlib.pyplot as plt


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.params = parameters

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        """ Predict using the traindata """

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regularizer': 'None', 'stepsize':0.01}

    def reset(self, parameters):
        self.params = parameters
        self.weights = None
        self.best_weight = None
        self.best_accuracy = None
        self.best_index = None
        self.accuracy_val = None
        self.accuracy_test = None

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """
        cost = 0.0
        E = self.sigmoid(np.dot(X,theta.T))
        cost = E - y

        return cost
    
    def sigmoid(self, xvec):
        """ Compute the sigmoid function """
        # Cap -xvec, to avoid overflow
        # Undeflow is okay, since it get set to zero
    
        vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))
     
        return vecsig    

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))
        cost = self.logit_cost(theta, X, y)
        grad = np.dot(cost, X)
        
        return grad

    def learn(self, Xtrain, ytrain, Xval, Yval, Xtest = None, Ytest = None):
        """
        Learn the weights using the training data
        """
        # set validation data
        self.Xtrain = Xtrain
        self.Ytrain = ytrain
        self.Xval = Xval
        self.Yval = Yval
        self.Xtest = Xtest
        self.Ytest = Ytest
        # init weights
        self.weights = np.random.rand(Xtrain.shape[1],)-0.5
        # variables could change
        stepsize = self.params['stepsize']
        num_steps = self.params['num_steps']
        batch_size = self.params['batch_size']
        batchs = int(Xtrain.shape[0]/batch_size)
        self.accuracy_train = np.zeros(num_steps)
        self.accuracy_val = np.zeros(num_steps)
        self.accuracy_test = np.zeros(num_steps)
        # learn weight
        for i in range(num_steps):
            for j in range(batchs):
                grad = self.logit_cost_grad(self.weights, Xtrain[j*batch_size:(j+1)*batch_size], ytrain[j*batch_size:(j+1)*batch_size])
                self.weights -= stepsize*grad/batch_size
            val_accuracy, train_accuracy = self.update_best_weight(i)
            if i%10 == 0 or True:
                self.save_accuracy(val_accuracy, train_accuracy, i)
        # save data into file
        self.best_accuracy = self.accuracy_test[self.best_index]
        pickle.dump( 
            (self.accuracy_val, self.accuracy_test, self.accuracy_train, self.best_accuracy, self.best_weight), 
            open("learning_acc.pkl", "wb")
        )
        print("---------------------------------------------------------")
        #print(self.accuracy_val, self.accuracy_test, self.accuracy_train, self.best_accuracy, self.best_weight)
        return self.best_weight, self.best_accuracy

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)
        ytest = utils.threshold_probs(utils.sigmoid(np.dot(Xtest, self.weights.T)))
        # for test
        #print(self.weights)

        assert len(ytest) == Xtest.shape[0]
        return ytest
    
    def update_best_weight(self, index):
        """
        Use new learned weight to check best parameter
        """        
        # first time : set weight and accuracy as current
        if self.best_weight is None:
            self.best_weight = self.weights
            pred = self.predict(self.Xval)
            acc_val = utils.getaccuracy(pred, self.Yval)    
            self.best_accuracy = acc_val
        # update the best weight & accuracy
        else:
            pred = self.predict(self.Xval)
            acc_val = utils.getaccuracy(pred, self.Yval)
            if acc_val > self.best_accuracy:
                self.best_accuracy = acc_val
                self.best_index = index
                self.best_weight = self.weights
                
        # get train accuracy
        pred_train = self.predict(self.Xtrain)
        acc_train = utils.getaccuracy(pred_train, self.Ytrain)
        
        return acc_val, acc_train
                
    def save_accuracy(self, val_accuracy, train_accuracy, index):
        """
        Save accuracy info for training graph
        """ 
        # validation accuracy
        self.accuracy_val[index] = val_accuracy
        self.accuracy_train[index] = train_accuracy
        
        # test accuracy
        test = True
        if test:
            pred = self.predict(self.Xtest)
            test_accuracy = utils.getaccuracy(pred, self.Ytest)
            self.accuracy_test[index] = test_accuracy
            