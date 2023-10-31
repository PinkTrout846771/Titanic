# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:52:11 2022

@author: 1002532
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

def readDataFile():
    train_fileName = 'TitanicSurvival_Train.csv'
    test_fileName = 'TitanicSurvival_Test.csv'
    train_data = open(train_fileName, 'rt')
    test_data = open(test_fileName, 'rt')
    #loadtxt defaults to floats
    train_data = np.loadtxt(train_data, usecols = (1,2,5,6,7,8,10), skiprows = 1, delimiter=",", dtype = np.str)
    test_data = np.loadtxt(test_data, usecols = (0,1,4,5,6,7,9), skiprows = 1, delimiter=",", dtype = np.str)
     
    train_data[:, 2][(train_data[:, 2] == 'female')] = 1
    train_data[:, 2][(train_data[:, 2] == 'male')] = 0
    test_data[:, 2][(test_data[:, 2] == 'female')] = 1
    test_data[:, 2][(test_data[:, 2] == 'male')] = 0
    
    pClassColms_train = train_data[:, 1:2]
    ohe = OneHotEncoder(categories='auto')
    passClassColms_train = ohe.fit_transform(pClassColms_train).toarray().astype(np.float64)
    
    pClassColms_test = test_data[:, 1:2]
    ohe = OneHotEncoder(categories='auto')
    passClassColms_test = ohe.fit_transform(pClassColms_test).toarray().astype(np.float64)
    
    #print(str(train_data[:, 2]))
    train_data[:, 3][train_data[:, 3] == ""] = np.nan
    x = train_data[:, 3].astype(np.float64)
    saved_mean = np.nanmean(x[:])
    train_data[:, 3][np.isnan(x[:])] = saved_mean
    
    test_data[:, 3][test_data[:, 3] == ""] = np.nan
    y = test_data[:, 3].astype(np.float64)
    saved_mean = np.nanmean(y[:])
    test_data[:, 3][np.isnan(y[:])] = saved_mean
    
    test_data[:, 6][test_data[:, 6] == ""] = np.nan
    z = test_data[:, 6].astype(np.float64)
    saved_mean = np.nanmean(z[:])
    test_data[:, 6][np.isnan(z[:])] = saved_mean
    #test_data[:,1:], test_data[:,0] 
    
    
    
    return passClassColms_train, passClassColms_test, train_data[:,1:], train_data[:,0], test_data[:,1:], test_data[:,0]
    

pClassTrain, pClassTest, X_train, Y_train, X_test, passId = readDataFile()
#X = np.column_stack((bias, X))

X_test = X_test.astype(float)
X_train = X_train.astype(float)
Y_train = Y_train.astype(float)

Y_train = Y_train.reshape(len(Y_train), 1)
print("Sample X_train row 1 before standardization: \n" + str(X_train[0]))


mean = np.mean(X_train, axis = 0)
std = np.std(X_train, axis = 0)
print("\nMean: \n" + str(mean))
print("\nStd: \n" + str(std))

X_train = (X_train - mean)/std



print("\nSample X_train row 1 after standardization: \n" + str(X_train[0]))

bias = np.ones((len(X_train),1))
X_train = np.concatenate((bias, X_train), axis=1)
X_train = np.concatenate((pClassTrain, X_train), axis=1)

og_weights = np.array([0,0,0,0,0,0,0,0,0,0]).reshape(10,1)

def activation(X, W):
  return 1/(1+np.exp(-np.dot(X, W)))

def calcCost(X,W,Y):
  a = Y * np.log(activation(X, W))
  b = (1-Y) * np.log(1 - activation(X, W))
  return (-1/len(X))*np.sum(a+b)
    

def calcGradient(X,W,Y):
    return np.dot(X.T, activation(X, W) - Y)/len(X)


LR = 0
iterations = 1
max_iterations = 100
min_costs = []
for i in range(1, 1001):
    iterations = 1
    LR += 0.001
    new_costs = []
    weights = og_weights
    while iterations < max_iterations:
        new_costs.append(calcCost(X_train, weights, Y_train))
        grad = calcGradient(X_train, weights, Y_train)
        weights =  weights - (LR*grad)
        #vect_diff = np.linalg.norm(grad)
        iterations += 1
    min_costs.append(min(new_costs))
    
#print(str(min(costs)) + ", " + str(costs.index(min(costs))))

print("\nMinimum cost after 100 iterations: " + str(min(min_costs)))
print("\nIndex: " + str(min_costs.index(min(min_costs))))
LR = min_costs.index(min(min_costs))/1000
print("\nBest Learning Rate: " + str(LR))
#ax1.plot(X_train[:,1], Y_train, 'rx')


#BEST LR IS 0.99

iterations = 1
best_costs = []
max_iterations = 21753
weights = np.array(np.zeros(10)).reshape(10, 1)

while iterations < max_iterations:
    best_costs.append(calcCost(X_train, weights, Y_train))
    grad = calcGradient(X_train, weights, Y_train)
    weights =  weights - (LR*grad)
    iterations += 1

fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(best_costs)
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Cost")
ax1.set_title("Cost plot")

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])

#print(activation(X_test, weights))

def predictTestData(x, w, mean, std):

    # 1) Write code to make a prediction for each row of x's
  #x = (x - mean)/std
  minx = np.min(x, axis=0)
  maxx = np.max(x, axis=0)

  x = (x-minx)/(maxx-minx)
  #print(x)
  #print(x[0,0]*w[0] + x[0,1]*w[1] + x[0,2]*w[2])
  x = np.concatenate((np.ones((len(x),1)), x), axis=1)
  x = np.concatenate((pClassTest, x), axis=1)
  pred = activation(x, w)
  pred[:,:][pred[:,:] >= 0.5] = 1
  pred[:,:][pred[:,:] < 0.5] = 0
    # NOTE: the passed in x matrix does not currently have the bias (x0) column
    # 2) Print the prediction and how far the prediction differs from the actual (Y)
    # 3) Return the prediction matrix
  #print("\nPredictions: \n" + str(pred))
  predictMatrix = pred
  return predictMatrix

preds = predictTestData(X_test, weights, mean, std)
passId = passId.astype(float).reshape(len(passId), 1)
preds = np.concatenate((passId, preds), axis=1)
np.savetxt('titanicPredict.csv', preds, fmt='%u',delimiter=',')


print("\n-------------------------------------------------------\n")
print("Percentage survived in the train data: %5.2f%%" % ((np.count_nonzero(Y_train == 1)/len(Y_train))*100))
print("Predicted percentage survived in the test data: %5.2f%%" % ((np.count_nonzero(preds[:, 1] == 1)/len(preds))*100))

train_preds = activation(X_train, weights)
train_preds[:,:][train_preds[:,:] >= 0.5] = 1
train_preds[:,:][train_preds[:,:] < 0.5] = 0

act_pred = np.concatenate((Y_train, train_preds), axis=1)

cnts = [0,0,0,0]
for i in range(len(act_pred)):
    if (act_pred[i, 0] == 0 and act_pred[i, 1] == 0):
        cnts[0] += 1
    elif (act_pred[i, 0] == 0 and act_pred[i, 1] == 1):
        cnts[1] += 1
    elif (act_pred[i, 1] == 0 and act_pred[i, 1] == 0):
        cnts[2] += 1
    else:
        cnts[3] += 1
        

print("\nConfusion Matrix - who survived?")
print("----------  ------------  -------------")
print("n = %3d     Predicted No  Predicted Yes" % (len(X_train)))
print("Actual No   %3d           %2d" % (cnts[0], cnts[1]))
print("Actual Yes  %3d           %3d" % (cnts[2], cnts[3]))
print("-----------  --------")
print("Accuracy:    %7.5f" % ((cnts[0] + cnts[3])/ len(X_train)))
print("Error Rate:  %7.5f" % (1-(cnts[0] + cnts[3])/ len(X_train)))
precision = cnts[3]/(cnts[3]+cnts[1])
print("Precision:   %8.6f" % (precision))
recall = cnts[3]/(cnts[3]+cnts[2])
print("Recall:      %7.5f" % (recall))
print("F1 Score:    %8.6f" % ((2*precision*recall)/(precision+recall)))
print("-----------  --------")