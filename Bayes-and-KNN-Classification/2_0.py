'''
Question 2.0

load the data and plot the means for each of the digit classes.
'''

import data
import numpy as np
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    means = np.zeros((10,64))
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        means[i,:] = np.mean(i_digits, axis = 0)
        # Compute mean of class i
    plt.figure(figsize = (20,5))
    num = means.shape[0]
    for i in range(num):
        plt.subplot(2,5,i+1)
        plt.imshow(means[i,:].reshape((8,8)),cmap = 'gray')
    plt.tight_layout()
    plt.show()

def main():
    train_data, train_labels, _, _ = data.load_all_data_from_zip('a2digits.zip', 'data')
    plot_means(train_data, train_labels)

if __name__ == '__main__':
    main()
