import pandas as pd
from plotnine import ggplot, geom_boxplot, aes
import numpy as np
#pandas.__version__

npf = pd.read_csv("npf_train.csv")
npf_test_hidden = pd.read_csv("npf_test_hidden.csv")

#(ggplot(npf,aes(x="class4", y="RHIRGA84.mean")) + geom_boxplot())

npf = npf.set_index("date")
npf = npf.drop("id", axis=1)
npf = npf.drop("partlybad", axis=1)
class2 = np.array(["nonevent","event"])
npf["class2"] = class2[(npf["class4"]!="nonevent").astype(int)]
#classf = np.array([-1,1])
#npf["classf"] = classf[(npf["class4"]!="nonevent").astype(int)]


import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


seed = 99

kernel = 1 * RBF(5)

models_text = ["KNeighbours5", "Gaussian","logisticRegression", "linearSVC", "SVC1", "SVC2", "RandomForest"]

models = [KNeighborsClassifier(5),
          GaussianProcessClassifier(kernel=kernel, random_state=seed, n_jobs=4),
          LogisticRegression(random_state=seed, solver='lbfgs', multi_class='ovr'), 
          LinearSVC(max_iter=10000),
          SVC(kernel="linear", C=0.025),
          SVC(gamma=2, C=1),
          RandomForestClassifier(n_estimators=100, max_depth=2, random_state=seed),
         ]

res = pd.DataFrame(index=["dummy", "OLS"])

#def loss(X_tr, y_tr, X_te, y_te, m):
#    return mean_squared_error(y_te, m.fit(X_tr, y_tr).predict(X_te), squared=False)

def loss(X_tr, y_tr, X_te, y_te, m):
    return log_loss()

#npf, _ = train_test_split(
#    npf, train_size=1000, random_state=42
#)
classf = np.array([-1,1])

X = npf.drop(["class2", "class4"], axis=1)
y = npf["class2"]
y_f = classf[(y !="nonevent").astype(int)]


headers = list(X.columns.values)
feature_names = np.array(headers)

# --- Sequential Feature selection using ridgeCV ------
# https://scikit-learn.org/stable/modules/feature_selection.html
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html

ridge = RidgeCV(alphas=np.logspace(-6,6, num=5)) #.fit(X, y_f) # use train data or total?

sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=10, direction="forward"
)

sfs_forward.fit(X,y_f)
#print(sfs_forward.get_support())
best_features=feature_names[sfs_forward.get_support()]
print(F"Best 10 features: {best_features}")
print("\n")

X = X[best_features]
# -----------------------------------------------------

X = StandardScaler().fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=math.floor(len(npf.index)*0.8), random_state=seed, shuffle=True, stratify=npf['class4']
)


for i, model in enumerate(models):
    model.fit(X_train, y_train)
    
    print(F"[{models_text[i]}]: ")
    print(F"  Validation score: {round(model.score(X_valid,y_valid), 4)}")
    cv_scores = cross_validate(model, X_train, y_train, scoring='accuracy', cv=5)
    print(F"  CV_score: {np.mean(cv_scores['test_score'])}")
    
    print("\n")
    

# train on test+validation data to produce final model
