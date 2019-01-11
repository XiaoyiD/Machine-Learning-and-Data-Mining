from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features

def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.plot(X[:,i],y,'.')
        plt.xlabel(features[i])
        plt.ylabel('y')
        # Plot feature i against y
    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    IdentitySize = X.shape[1]
    IdentityMatrix = np.zeros((IdentitySize, IdentitySize))
    np.fill_diagonal(IdentityMatrix, 1)
    lamb = 1
    XtX_lamb = np.dot(X.T,X) + lamb * IdentityMatrix
    XtY = np.dot(X.T,Y)
    w = np.linalg.solve(XtX_lamb, XtY)
    # implement linear regression
    return w

def SLICE(X,y):
    choice = np.random.choice(X.shape[0],X.shape[0]*8//10,replace = False)
    choice.sort()
    trainset = np.zeros(shape = (choice.shape[0],X.shape[1]))
    y_train = np.zeros(shape = (choice.shape[0]))
    testset = np.zeros(shape = (X.shape[0]-choice.shape[0],X.shape[1]))
    y_test = np.zeros(shape = (X.shape[0]-choice.shape[0]))
    r,t = 0,0
    for i in range(X.shape[0]):
        if i in choice:
            trainset[r,:]= X[i,:]
            y_train[r,]= y[i,]
            r=r+1
        else:
            testset[t,:]= X[i,:]
            y_test[t,]=y[i,]
            t = t+1
    return trainset,y_train,testset,y_test

def Result(X,y,w):
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    fitted_v = np.dot(X,w)
    MSE = np.mean((np.dot(X,w)-y)**2)
    return fitted_v,MSE

def EMM(X,y,w):
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    MASE = np.mean(np.fabs((np.dot(X,w)-y)))
    MAPE = np.mean(np.fabs((np.dot(X,w)-y)/y))
    return MASE,MAPE

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    # Information about the dataset
    print('\n',"The number of the dataset is:",X.shape[0])
    print('\n',"The dimension of the dataset is:",(X.shape[0],X.shape[1]+1))
    print('\n',"The number of the target is:",y.shape[0])
    print('\n',"The dimension of the target is:",y.shape)

    # Visualize the features
    visualize(X, y, features)

    # Split data into train and test
    trainset,y_train,testset,y_test = SLICE(X,y)

    # Fit regression model
    w = fit_regression(trainset, y_train)
    print(w,'\n')

    for i, tau in enumerate(features):
        print(features[i],':',w[i+1],'\n')

    # Compute fitted values, MSE, etc.
    fitted_v,MSE = Result(testset,y_test,w)
    print('\nMean square error : ',MSE)

    # two more error measurement metrics
    MASE, MAPE = EMM(testset,y_test,w)
    print('\nMean absolute scaled error : ',MASE)
    print('\nMean absolute percentage error : ',MAPE)

if __name__ == "__main__":
    main()
