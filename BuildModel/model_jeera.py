from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv(r"../Data/SteamJeeraMasino.csv")

df.drop(df.columns[[6, 7]], axis=1, inplace=True)

grade = df['Grade']
df['Grade'] = df['Grade'].replace({'A': 0, 'B': 1, 'C': 2})

X = df.drop('Grade', axis=1)
Y = df['Grade']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

logmodel = LogisticRegression()

logmodel.fit(X_Train, Y_Train)


# saving logmodel to disk
pickle.dump(logmodel, open(
    '../models/SteamJeeraMasino/model_logistic.pkl', 'wb'))

# loading model to compare the results
model = pickle.load(
    open('../models/SteamJeeraMasino/model_logistic.pkl', 'rb'))
print(model.predict([[12.0, 0, 3.9, 1.5, 0.5, 0.5]]))
print(model.predict([[12.2, 0, 4.2, 4.5, 0.5, 0.8]]))
print(model.predict([[13.5, 0.1, 4.8, 4.8, 0.8, 0.5]]))


gnb = GaussianNB()
gnb.fit(X_Train, Y_Train)
NB_Pred = gnb.predict(X_Test)
# saving gaussian model to disk
pickle.dump(gnb, open('../models/SteamJeeraMasino/model_nb.pkl', 'wb'))
naive = pickle.load(open('../models/SteamJeeraMasino/model_nb.pkl', 'rb'))
print(naive.predict([[12.0, 0, 3.9, 1.5, 0.5, 0.5]]))
print(naive.predict([[12.2, 0, 4.2, 4.5, 0.5, 0.8]]))
print(naive.predict([[13.5, 0.1, 4.8, 4.8, 0.8, 0.5]]))


svm_lin = SVC(kernel="linear", C=1)
svm_lin.fit(X_Train, Y_Train)
SVM_Pred = svm_lin.predict(X_Test)
# saving svm model to disk
pickle.dump(svm_lin, open('../models/SteamJeeraMasino/model_svm.pkl', 'wb'))

svm = pickle.load(open('../models/SteamJeeraMasino/model_svm.pkl', 'rb'))
print(svm.predict([[12.0, 0, 3.9, 1.5, 0.5, 0.5]]))
print(svm.predict([[12.2, 0, 4.2, 4.5, 0.5, 0.8]]))
print(svm.predict([[13.5, 0.1, 4.8, 4.8, 0.8, 0.5]]))


def acc_jeera():
    print('Accuracy Score: with Logistic Regression for Jeera Masino')
    print("{:.2%}".format(logmodel.score(X_Test, Y_Test)))
    print('')
    print('Accuracy Score: with Naive Bayes for Jeera Masino')
    print("{:.2%}".format(metrics.accuracy_score(Y_Test, NB_Pred)))
    print('')
    print('Accuracy Score: with SVM for Jeera Masino')
    print("{:.2%}".format(metrics.accuracy_score(Y_Test, SVM_Pred)))
