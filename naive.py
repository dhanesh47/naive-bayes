import csv
import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

model = GaussianNB()

def load_data():
    data = []
    f = open('C:/Users/HP/Downloads/spambase.data')
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        data.append(row)
    f.close()
    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data 
    return train_test_split(X, y, test_size=0.3, random_state=RandomState(2))

def main():
    X_train, X_test, y_train, y_test =load_data()
    
    print("Train data : ",len(X_train)," Test data : ",len(y_test))

    model.fit(X_train,y_train)

    predicted= model.predict( X_test )

    print("Accuracy:",metrics.accuracy_score( y_test, predicted ))


main()







































































