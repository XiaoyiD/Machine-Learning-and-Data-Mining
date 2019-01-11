'''
Question 1

'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer(stop_words='english')
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    # feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))
    return model

def SVM(tf_idf_train, train_labels, tf_idf_test, test_labels):
    from sklearn.svm import SVC
    svcclf = SVC(kernel='linear')
    parameters = { 'C': [1, 10]}
    gridSVC = GridSearchCV(svcclf, parameters, cv=10, scoring='accuracy', verbose=1,n_jobs= -1)
    gridSVC.fit(tf_idf_train, train_labels)
    print('best score is:', str(gridSVC.best_score_))
    print('best params are:', str(gridSVC.best_params_))
    Op_SVC=gridSVC.best_estimator_

    Op_SVC.fit(tf_idf_train, train_labels)
    train_pred = Op_SVC.predict(tf_idf_train)
    print('SVC train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = Op_SVC.predict(tf_idf_test)
    accuracy=(test_pred == test_labels).mean()
    print('SVC test accuracy = {}'.format((test_pred == test_labels).mean()))
    return Op_SVC,accuracy

def Op_SVM(tf_idf_train, train_labels, tf_idf_test, test_labels):
    from sklearn.svm import SVC
    Op_SVC=SVC(kernel='linear',C = 1)
    Op_SVC.fit(tf_idf_train, train_labels)

    train_pred = Op_SVC.predict(tf_idf_train)
    print('SVC train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = Op_SVC.predict(tf_idf_test)
    accuracy=(test_pred == test_labels).mean()
    print('SVC test accuracy = {}'.format((test_pred == test_labels).mean()))
    return Op_SVC,accuracy

def RFCCLF(tf_idf_train, train_labels, tf_idf_test, test_labels):
    from sklearn.ensemble import RandomForestClassifier
    rfcclf = RandomForestClassifier(n_jobs=-1,max_features='auto')
    E = list(range(80,100,5))
    D = list(range(30,50,5))
    parameters = {'n_estimators': E,'max_depth': D,'min_samples_leaf': [1, 2, 4]}
    gridRFC = GridSearchCV(rfcclf, parameters, cv=10, scoring='accuracy', verbose=1,n_jobs= -1)
    gridRFC.fit(tf_idf_train, train_labels)
    print('best score is:', str(gridRFC.best_score_))
    print('best params are:', str(gridRFC.best_params_))
    Op_RFC=gridRFC.best_estimator_

    Op_RFC.fit(tf_idf_train, train_labels)
    # hyper-param tuning
    train_pred = Op_RFC.predict(tf_idf_train)
    print('RandomForestClassifier train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = Op_RFC.predict(tf_idf_test)
    accuracy=(test_pred == test_labels).mean()
    print('RandomForestClassifier test accuracy = {}'.format((test_pred == test_labels).mean()))
    return Op_RFC,accuracy

def Op_RFCCLF(tf_idf_train, train_labels, tf_idf_test, test_labels):
    from sklearn.ensemble import RandomForestClassifier
    Op_RFC = RandomForestClassifier(n_estimators=85,max_depth=45,min_samples_leaf=2,n_jobs=-1,max_features='auto')
    Op_RFC.fit(tf_idf_train, train_labels)

    train_pred = Op_RFC.predict(tf_idf_train)
    print('RandomForestClassifier train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = Op_RFC.predict(tf_idf_test)
    accuracy=(test_pred == test_labels).mean()
    print('RandomForestClassifier test accuracy = {}'.format((test_pred == test_labels).mean()))
    return Op_RFC,accuracy

def MNBCLF(tf_idf_train, train_labels, tf_idf_test, test_labels):
    from sklearn.naive_bayes import MultinomialNB
    mnbclf = MultinomialNB()
    alpha = np.linspace(1e-4,1,500)
    param_gridmnb = dict(alpha=alpha,fit_prior= [True,False])
    gridMNB = GridSearchCV(mnbclf, param_gridmnb, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
    gridMNB.fit(tf_idf_train, train_labels)
    print('best score is:', str(gridMNB.best_score_))
    print('best params are:', str(gridMNB.best_params_))
    Op_MNB = gridMNB.best_estimator_

    Op_MNB.fit(tf_idf_train, train_labels)
    train_pred = Op_MNB.predict(tf_idf_train)
    print('MultinomialNB train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = Op_MNB.predict(tf_idf_test)
    accuracy=(test_pred == test_labels).mean()
    print('MultinomialNB test accuracy = {}'.format((test_pred == test_labels).mean()))
    return Op_MNB,accuracy

def Op_MNBCLF(tf_idf_train,train_labels,tf_idf_test,test_labels):
    from sklearn.naive_bayes import MultinomialNB
    Op_MNB = MultinomialNB(alpha = 0.012122845691382765,fit_prior= False)
    Op_MNB.fit(tf_idf_train,train_labels)
    train_pred = Op_MNB.predict(tf_idf_train)
    print('MultinomialNB train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = Op_MNB.predict(tf_idf_test)
    accuracy = (test_pred == test_labels).mean()
    print('MultinomialNB test accuracy = {}'.format((test_pred == test_labels).mean()))
    return Op_MNB,accuracy

def confusion_matrix(tf_idf_test,test_labels,best_estimator):
    test_pred = best_estimator.predict(tf_idf_test)
    matrix = np.zeros((20,20))
    n = test_pred.shape[0]
    for t in range(n):
        j = test_labels[t]
        i = test_pred[t]
        matrix[i,j] = matrix[i,j]+1
    # print('Below is confusion matrix:')
    # print(matrix)
    return matrix

def most_confuse(matrix):
    n = matrix.shape[0]
    for i in range(n):
        matrix[i,i] = 0
    total = np.sum(matrix,axis = 0).reshape((1,n))
    normal = matrix/total
    normal = normal + normal.T
    re = np.where(normal == np.max(normal))
    print('This classifier is most confused about:',re[0]+1)
    return re

def main():
    train_data, test_data = load_data()

    tf_idf_train, tf_idf_test = tf_idf_features(train_data, test_data)
    bnb_model = bnb_baseline(tf_idf_train, train_data.target, tf_idf_test, test_data.target)

    ## You can uncomment the code below to see how hyper-parameters are tuned
    ## Note: This process may take more than 40 minutes.

    # SVM(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    # RFCCLF(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    # MNBCLF(tf_idf_train, train_data.target, tf_idf_test, test_data.target)

    accuracy = {}
    OP_RFC,acc1 = Op_RFCCLF(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    OP_MNB,acc2 = Op_MNBCLF(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    OP_SVM,acc3 = Op_SVM(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    accuracy[acc1] = ["RandomForestClassifier",OP_RFC]
    accuracy[acc2] = ["MultinomialNB",OP_MNB]
    accuracy[acc3] = ["SVM",OP_SVM]
    bestAccuracy = sorted(accuracy.keys(),reverse=True)[0]
    bestModel = accuracy[bestAccuracy][1]
    print('The best model is {}, and the corresponding accuracy is {}'.format(accuracy[bestAccuracy][0], bestAccuracy))
    matrix = confusion_matrix(tf_idf_test,test_data.target,bestModel)
    most_confuse(matrix)

if __name__ == '__main__':
    main()
