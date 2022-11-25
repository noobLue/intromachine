import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier

npf = pd.read_csv("npf_train.csv")
npf = npf.set_index("date")
npf = npf.drop("id",axis=1)

#print(npf.describe())
npf = npf.drop("partlybad",axis=1)
#print(npf)

class2 = np.array(["nonevent", "event"])
X = npf.drop("class4", axis=1)
npf["class2"] = class2[(npf["class4"]!="nonevent").astype(int)]
y = npf["class2"]
#print(list(npf.columns))
print(len(y))

# Gaussian Process Classifier
if 0:
    for kernel_width in [0.1,0.3,0.5,0.7,1,1.5,2]:
        accuracies = []
        for seed in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=300, random_state=seed, shuffle=True)

            kernel = kernel_width * RBF(5)
            gpc = GaussianProcessClassifier(kernel=kernel, random_state=seed, n_jobs=4).fit(X_train, y_train)
            y_predict = gpc.predict(X_test)
            accuracy = sum(np.array(y_predict==y_test).astype(int))/len(y_test)
            print(accuracy)
            accuracies.append(accuracy)

        print(f"mean (width={kernel_width}):", np.mean(accuracies))

# KNeighbors Classifier
if 1:
    for n in range(3,10):
        accuracies = []
        for seed in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=300, random_state=seed, shuffle=True)
            knc = KNeighborsClassifier(n).fit(X_train, y_train)
            y_predict = knc.predict(X_test)
            accuracy = sum(np.array(y_predict==y_test).astype(int))/len(y_test)
            print(accuracy)
            accuracies.append(accuracy)
        print(f"mean (neighbors={n}):", np.mean(accuracies))

