import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

BATCHES = 50

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
        if m is None:
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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    boston = load_boston()
    X = boston.data
    y = boston.target
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1.T))
    sum2 = np.sqrt(np.dot(vec2.T, vec2))
    return dot / (sum1 * sum2)


# Implement linear regression gradient
def lin_reg_gradient(X, y, w):
    de = y - np.dot(X, w)
    de.resize((de.shape[0], 1))
    grad = 2 * (de * X).mean(axis=0)
    grad.resize((1, grad.shape[0]))
    return grad

def Grad(X, y, w, batch):
    Grad = np.zeros((500, X.shape[1]))
    for i in range(500):
        # Create a batch sampler to generate random batches from data
        batch_sampler = BatchSampler(X, y, batch)
        # Example usage
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        Grad[i, :] = batch_grad
    return Grad

def visualize(X,M = None):
    M = np.arange(400)+1
    plt.figure(figsize=(20, 5))
    plt.plot(np.log(M),np.log(X[:,1]),'.')
    plt.xlabel('log(m)')
    plt.ylabel('log(var of w1)')
    plt.title('log σ1 against log m')
    plt.show()
def Var(Grad):
     var = np.zeros((1,Grad.shape[1]))
     for i in range(Grad.shape[1]):
         arg = np.mean(Grad[:,i])
         vari = np.sum((Grad[:,i]-arg)**2)/(Grad.shape[0]-1)
         var[0,i]=vari
     return var

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    grad = Grad(X,y,w,BATCHES)
    lin_grad = grad.mean(axis=0)
    lin_grad.resize((1, lin_grad.shape[0]))
    tr_grad = lin_reg_gradient(X, y, w)

    sq_s = np.sqrt(((lin_grad - tr_grad) ** 2).sum())
    cos_s = cosine_similarity(lin_grad, tr_grad.T)
    print(sq_s)
    print(cos_s)
    VAR = np.zeros((400,14))
    for Bat in range(400):
        VAR[Bat,:] = Var(Grad(X,y,w,Bat+1))
    visualize(VAR)

if __name__ == '__main__':
    main()
