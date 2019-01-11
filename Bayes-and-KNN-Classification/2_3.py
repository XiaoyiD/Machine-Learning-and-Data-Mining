'''
Question 2.3
Implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt
def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)
def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    Return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        eta[i,:] = (np.sum(i_digits,axis = 0)+2)/(i_digits.shape[0]+4)
    return eta

def plot_images(class_images,showtime = False):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    figure = plt.figure(figsize = (20,5))
    for i in range(10):
        img_i = class_images[i].reshape(8,8)
        # ...
        plt.subplot(2,5,i+1)
        plt.imshow(img_i,cmap = 'gray')
    plt.tight_layout()
    if showtime:
        timer = figure.canvas.new_timer(interval=5000)  # creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(plt.close)
        timer.start()
    plt.show()
    figure.savefig('n(k)')

def generate_new_data(eta):
    '''
    Sample a new data point from generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    plot_images(binarize_data(eta))

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Return an n x 10 numpy array
    '''
    Wc = np.zeros((10,64))
    wc = np.zeros((10,1))
    gene_lld = np.zeros((bin_digits.shape[0],10))
    for i in range(eta.shape[0]):
        wc[i] = np.sum(np.log(-1*eta[i]+1))
        for j in range(eta.shape[1]):
            Wc[i,j] = np.log(eta[i,j]/(1-eta[i,j]))
    for i in range(10):
        for j in range(bin_digits.shape[0]):
            gene_lld[j,i] = np.dot(bin_digits[j],Wc[i,:])+wc[i]
    return gene_lld


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    A numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    condi_likelihood = generative_likelihood(bin_digits,eta) + np.log(1/10)
    return condi_likelihood

def avg_conditional_likelihood(bin_digits, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    n = bin_digits.shape[0] // 10
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    true_cond_likelihood = np.zeros((n,10))
    for i in range(10):
        true_cond_likelihood[:,i]= cond_likelihood[n*i:n*(i+1),i]
    avg_cond = np.mean(true_cond_likelihood,axis = 0)
    # Compute as described above and return
    for i in range(avg_cond.shape[0]):
        print("The average conditional likelihood of data in class ", i+1 ," is :",avg_cond[i])
    print('\n', 'The average conditional likelihood over true class labels of all data is: ', np.average(avg_cond))
    # Compute as described above and return
    return avg_cond

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    classify_retult = np.argmax(cond_likelihood, axis = 1)
    return classify_retult

def accuracy(labels,bin_digits,eta):
    accu = np.equal(classify_data(bin_digits,eta),labels).mean()
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

    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    plot_images(eta)
    generative_likelihood(train_data, eta)

    print('The avg_condi_llh in train data:')
    avg_conditional_likelihood(train_data, eta)
    print('\nThe avg_condi_llh in test data: ')
    avg_conditional_likelihood(test_data, eta)
    print('\nThe accuracy for train data is: ', accuracy(train_labels, train_data, eta))
    print('The accuracy for test data is: ', accuracy(test_labels, test_data, eta))
    # Evaluation

    generate_new_data(eta)

if __name__ == '__main__':
    main()
