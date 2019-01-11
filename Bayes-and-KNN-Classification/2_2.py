'''
Question 2.2
Implement and evaluate the Conditional Gaussian classifier.
'''
import data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

Num = 7000
T_Num = 4000
Targ = 10
Features = 64

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class
    Return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((Targ, Features))
    train_data_i = np.zeros((Targ,train_data.shape[0]//Targ,Features))
    for i in range(Targ):
        q = np.argwhere(train_labels == i)
        q = np.squeeze(q)
        train_data_i[i,:,:] = train_data[q, :]
        means[i,:] = np.mean(train_data_i[i,:,:],axis=0)
    # Compute means
    return means, train_data_i

def compute_sigma_mles(train_data_i, means):
    '''
    Compute the covariance estimate for each digit class

    Return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((Targ,Features,Features))
    # Compute covariances
    for i in range(Targ):
        E = train_data_i[i,:,:].reshape(train_data_i.shape[1],Features) - means[i,:].reshape(1,-1)
        covariances[i,:,:] = np.dot(E.T,E)/train_data_i.shape[1]
        IdentityMatrix = np.zeros((Features,Features))
        np.fill_diagonal(IdentityMatrix, 1)
        lamb = 0.01
        covariances[i,:,:] += lamb * IdentityMatrix

    return covariances

def plot_cov_diagonal(covariances,showtime = True):
    # Plot the diagonal of each covariance matrix side by side
    figure = plt.figure(figsize = (20,5))
    for i in range(Targ):
        cov_diag = np.log(np.diag(covariances[i])).reshape(8,8)
        plt.subplot(2,5,i+1)
        plt.imshow(cov_diag,cmap = 'gray')
    plt.tight_layout()
    if showtime:
        timer = figure.canvas.new_timer(interval=5000)  # creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(plt.close)
        timer.start()
    plt.show()
    figure.savefig('covariances.png')

def plot_means(train_data, train_labels, showtime = True):
    means = np.zeros((Targ,train_data.shape[1]))
    for i in range(Targ):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        means[i,:] = np.mean(i_digits, axis = 0)
        # Compute mean of class i
    figure = plt.figure(figsize = (20,5))
    num = means.shape[0]
    for i in range(num):
        plt.subplot(2,5,i+1)
        plt.imshow(means[i,:].reshape((8,8)),cmap = 'gray')
    plt.tight_layout()
    if showtime:
        timer = figure.canvas.new_timer(interval=5000)  # creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(plt.close)
        timer.start()
    plt.show()
    figure.savefig("means.png")


def plot_egi(covariances, showtime = True):
    egi = np.zeros((Targ,covariances.shape[1]))
    for i in range(Targ):
        egi_va,egi_ve =np.linalg.eig(covariances[i].reshape((Features,Features)))
        egi[i,:] = egi_ve[np.argmax(egi_va),:]
        # Compute mean of class i
    figure = plt.figure(figsize = (20,5))

    for i in range(Targ):
        plt.subplot(2,5,i+1)
        plt.imshow(egi[i,:].reshape((8,8)),cmap = 'gray')
    plt.tight_layout()
    if showtime:
        timer = figure.canvas.new_timer(interval=5000)  # creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(plt.close)
        timer.start()
    plt.show()
    figure.savefig('egi.png')

def generative_likelihood(digits, means, covariances):

    generative_likelihood = np.zeros((digits.shape[0],Targ))
    for i in range(digits.shape[0]):
        for j in range(10):
            I = np.linalg.inv(covariances[j].reshape((Features,Features)))
            D = np.linalg.det(covariances[j, :, :])
            exp = np.exp(-0.5 * np.dot(np.dot((digits[i]-means[j]).T,I),(digits[i]-means[j])))
            generative_likelihood[i,j] = prob = ((2 * np.pi) ** (- Features / 2)) * (D ** (-1 / 2)) * exp
    return generative_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    A numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    generative_likelihoods = generative_likelihood(digits, means, covariances)
    conditional_likelihoods = generative_likelihoods/np.sum(generative_likelihoods, axis=1, keepdims=True)
    con = np.log(conditional_likelihoods)
    return con


def avg_conditional_likelihood(digits, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

    AVG( log p(y_i|x_i, mu, Sigma) )
    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    n = digits.shape[0]//Targ
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    true_cond_likelihood = np.zeros((n,Targ))
    for i in range(10):
        true_cond_likelihood[:,i]= cond_likelihood[n*i:n*(i+1),i]
    avg_cond = np.mean(true_cond_likelihood,axis = 0)
    # Compute as described above and return

    print('The average conditional likelihood over true class labels of all data is: ',np.average(avg_cond))
    return avg_cond

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits,means,covariances)
    # Compute and return the most likely class
    classify_retult = np.argmax(cond_likelihood, axis = 1)
    return classify_retult

def accuracy(labels,digits,means,covariance):
    accu = np.equal(classify_data(digits,means,covariance),labels).mean()
    return accu

def Order(train_data,train_labels):
    train_data_new = data.get_digits_by_label(train_data, train_labels, 0)
    train_labels_new = np.zeros((train_data.shape[0]//10,1))
    for i in range(1, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        train_data_new = np.vstack((train_data_new,i_digits))
        i_labels = np.zeros((train_data.shape[0]//10,1))+ i
        train_labels_new = np.vstack((train_labels_new,i_labels))
    train_data = train_data_new
    train_labels = np.squeeze(train_labels_new)
    return train_data,train_labels

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, train_labels = Order(train_data,train_labels)
    test_data , test_labels = Order(test_data,test_labels)
    # Fit the model
    means,train_data_i = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data_i, means)
    plot_cov_diagonal(covariances)
    plot_means(train_data, train_labels)
    plot_egi(covariances)
    # Evaluation
    print('The avg_condi_llh in train data:')
    avg_conditional_likelihood(train_data,means,covariances)
    print('\nThe avg_condi_llh in test data: ')
    avg_conditional_likelihood(test_data,means,covariances)
    print('\nThe accuracy for train data is: ',accuracy(train_labels, train_data, means, covariances))
    print('The accuracy for test data is: ',accuracy(test_labels, test_data, means, covariances))

if __name__ == '__main__':
    main()
