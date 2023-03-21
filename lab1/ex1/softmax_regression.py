# coding=utf-8
import numpy as np

def softmax(scores):
    sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax = np.exp(scores)/sum_exp
    return softmax

def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    # x:[m,n], y:[k,m], theta[k,n]
    n_samples, n_features = x.shape
    n_classes = y.shape[0]
    lam = 0.01 #正则项
    for i in range(iters):
        scores = np.dot(x, theta.T)
        probs = softmax(scores)
        loss = - (1.0 / n_samples) * np.sum(y.T * np.log(probs))
        print("第",i,"次loss:%f" % (loss))
        dw = -(1.0 / n_samples) * np.dot((y - probs.T), x) + lam * theta
        # 更新权重矩阵
        theta = theta - alpha * dw

    return theta
    
