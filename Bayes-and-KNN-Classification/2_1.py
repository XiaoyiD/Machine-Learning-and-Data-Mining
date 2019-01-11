'''
Question 2.1

Implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        '''
        dist = self.l2_distance(test_point)
        index = dist.argsort()
        targets = np.zeros((1, k))
        dist = np.sort(dist)

        for i in range(k):
            targets[0, i] = self.train_labels[index[i]]
        targets = targets.astype((int))
        if targets.size == 1:
            digit = targets[0, 0]
        else:
            targets = np.squeeze(targets)
            b = np.bincount(targets)
            candi = np.zeros((3, 10))
            freq = np.max(b)
            j = 0
            for i in range(b.shape[0]):
                if b[i] >= freq:
                    candi[0, j] = b[i]
                    candi[1, j] = i
                    j = j + 1
                else:
                    pass
            for i in range(10):
                if candi[0, i] != 0:
                    targ = candi[1, i]
                    for j in range(k):
                        if targets[j] == targ:
                            candi[2, i] = candi[2, i] + dist[j]
                else:
                    break
            c = np.argmin(candi[2,0:i])
            digit = candi[1, c]
        return digit

def run_k_fold(knn,kf, k_range = np.arange(1,16)):
    x = knn.train_data
    x_labels = knn.train_labels
    N = x.shape[0]
    dim = x.shape[1]

    idx_train = np.zeros((N - N//kf,1))
    idx_test = np.zeros((N//kf,1))
    idx = np.random.permutation(range(N))
    accuracy = np.zeros(shape = (kf,15))
    n = N // kf
    for k in k_range:
        for i in range(kf):
            idx_test = idx[(i * n):((i + 1) * n)]
            idx_train = np.concatenate((idx[0:i * n], idx[((i + 1) * n):(-1)]))
            x_test = x[idx_test,:]
            x_train = x[idx_train,:]
            x_test_labels = x_labels[idx_test]
            x_train_labels = x_labels[idx_train]
            knn_v = KNearestNeighbor(x_train,x_train_labels)
            accuracy[i,k-1] = classification_accuracy(knn_v,k,x_test,x_test_labels)
    return accuracy
def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    cheque = np.zeros(eval_data.shape[0])
    for i in range(eval_data.shape[0]):
        cheque[i] = (knn.query_knn(eval_data[i], k) == eval_labels[i])
    return cheque.mean()

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    train_accu_1 = classification_accuracy(knn, 1, train_data, train_labels)
    train_accu_15 = classification_accuracy(knn, 15, train_data, train_labels)
    test_accu_1 = classification_accuracy(knn, 1, test_data, test_labels)
    test_accu_15 = classification_accuracy(knn, 15, test_data, test_labels)
    print('For K=1,the train classification accuracy is:', train_accu_1)
    print('For K=15,the train classification accuracy is:', train_accu_15)
    print('For K=1,the test classification accuracy is:', test_accu_1)
    print('For K=15,the test classification accuracy is:', test_accu_15)

    acc = run_k_fold(knn,10, k_range=np.arange(1, 16))
    acc_k = np.mean(acc,axis = 0)
    opt_k = np.argmax(acc_k)+1

    print('The optimal K from cross-validation is :',opt_k)
    print('The train classification accuracy of optimal K:',classification_accuracy(knn, opt_k, train_data,train_labels))
    print('The average accuracy across folds of optimal K:',np.max(acc_k))

    Test_accu = classification_accuracy(knn, opt_k, test_data, test_labels)
    print('The test accuracy with optical K is :',Test_accu)


if __name__ == '__main__':
    main()
