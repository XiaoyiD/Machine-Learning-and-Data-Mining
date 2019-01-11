import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)
class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    '''
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.
        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if not m:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

class GradientDescent(object):
    '''
    A gradient descent optimizer with momentum
    '''
    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta

    def update_params(self, params, grad, temp):
        # Update parameters using GD with momentum and return
        # the updated parameters
        temp_new = (-1) * self.lr * grad + self.beta * temp
        params_new = params + temp_new
        return params_new,temp_new

class SVM(object):
    '''
    A Support Vector Machine
    '''
    def __init__(self, c, feature_count):
        self.c = c

    def hinge_loss(self, X, y, w):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        n = X.shape[0]
        m = X.shape[1]
        loss = np.zeros(n)

        for i in range(n):
            value = 1 - y[i] * np.dot(w.reshape(1,m),X[i])
            loss[i] = max(value,0)
        sum_loss = 0.5 * np.square(w[1:,]).sum() + self.c * loss.sum() / n
        return sum_loss

    def grad(self, X, y, w):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Return the gradient with respect to the SVM parameters (shape (m,)).
        '''
        n = X.shape[0]
        m = X.shape[1]
        sum = np.zeros(m)

        for i in range(n):
            value = 1 - y[i] * np.dot(w.reshape(1,m),X[i])
            if  value > 0:
                sum = sum - y[i] * X[i]
        grad = w + self.c / n * sum
        return grad

    def classify(self, X, w):
        '''
        Classify new input data matrix (shape (n,m)).

        Return the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        n = X.shape[0]
        m = X.shape[1]
        result = np.zeros(n)
        for i in range(n):
            result[i] = 1 if np.dot(w.reshape(1,m),X[i])> 0 else -1
        return result

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x
    def func_grad(x):
        return 0.02 * x
    w = w_init
    w_history = [w_init]
    temp = np.zeros_like(w_init)
    for i in range(steps):
        # Optimize and update the history
        grad = func_grad(w)
        w,temp = optimizer.update_params(w,grad,temp)
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    n = train_data.shape[0]
    m = train_data.shape[1]
    svm = SVM(penalty,m)
    w = np.random.normal(0.0, 0.1, m)
    temp = np.zeros_like(w)
    batch = BatchSampler(train_data,train_targets,batchsize)
    for i in range(iters):
        train_data_r, train_targets_r = batch.get_batch(batchsize)
        grad = svm.grad(train_data_r,train_targets_r,w)
        w,temp = optimizer.update_params(w,grad,temp)
    loss = svm.hinge_loss(train_data,train_targets,w)
    result = svm.classify(train_data,w)
    accuracy = np.equal(result,train_targets).mean()
    return w , accuracy, loss


if __name__ == '__main__':
    optimizer = GradientDescent(1.0, 0.9)
    w_history = optimize_test_function(optimizer)
    optimizer2 = GradientDescent(1)
    w_history2 = optimize_test_function(optimizer2)
    plt.figure(figsize=(20, 5))
    plot1 = plt.plot(w_history,'.')
    plot2 = plt.plot(w_history2,'.')
    plt.xlabel('iterations')
    plt.ylabel('Params')
    plt.show()

    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.column_stack((np.ones((train_data.shape[0], 1)), train_data))
    test_data = np.column_stack((np.ones((test_data.shape[0], 1)), test_data))
    optimizer = GradientDescent(0.05, 0.1)
    w, train_accuracy, train_loss = optimize_svm(train_data, train_targets,1.0,optimizer,100,500)

    test_svm = SVM(1.0,test_data.shape[1])
    test_result = test_svm.classify(test_data,w)
    test_loss = test_svm.hinge_loss(test_data,test_result,w)
    test_accuracy = np.equal(test_targets,test_result).mean()
    print('The classification accuracy on the training set with beta=1 is:',train_accuracy)
    print('The loss of the training set is:',train_loss)
    print('The classification accuracy on the testing set with beta=1 is:',test_accuracy)
    print('The loss of the testing set is:',test_loss)

    optimizer2 = GradientDescent(0.05,0)
    w2, train_accuracy2, train_loss2 = optimize_svm(train_data, train_targets,1.0,optimizer2,100,500)

    test_result2 = test_svm.classify(test_data,w2)
    test_loss2 = test_svm.hinge_loss(test_data,test_result2,w2)
    test_accuracy2 = np.equal(test_targets,test_result2).mean()
    print('The classification accuracy on the training set with beta=0 is:',train_accuracy2)
    print('The loss of the training set is:',train_loss2)
    print('The classification accuracy on the testing set with beta=0 is:',test_accuracy2)
    print('The loss of the testing set is:',test_loss2)

    plt.figure(figsize = (20,5))
    plt.subplot(1,2,1)
    plt.imshow(w[1:,].reshape((28,28)),cmap = 'gray')
    plt.subplot(1,2,2)
    plt.imshow(w2[1:,].reshape((28,28)),cmap = 'gray')
    plt.tight_layout()
    plt.show()
