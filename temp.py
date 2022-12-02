import pandas as pd
from plotnine import ggplot, geom_boxplot, aes
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# --- Data preprocessing ---------------------------------------------

npf = pd.read_csv("npf_train.csv")
npf_test_hidden = pd.read_csv("npf_test_hidden.csv")
#print(npf.groupby('class4').size())

#npf = npf.set_index("date")
npf = npf.drop("date", axis=1)
npf = npf.drop("id", axis=1)
npf = npf.drop("partlybad", axis=1)
class2 = np.array(["nonevent","event"])
npf["class2"] = class2[(npf["class4"]!="nonevent").astype(int)]

#npf_test_hidden = npf_test_hidden.set_index("date")
npf_test_hidden = npf_test_hidden.drop("date", axis=1)
npf_test_hidden = npf_test_hidden.drop("id", axis=1)
npf_test_hidden = npf_test_hidden.drop("partlybad", axis=1)
npf_test_hidden = npf_test_hidden.drop("class4", axis=1)

seed = 99

classf = np.array([-1,1])

X_test = npf_test_hidden
X = npf.drop(["class2", "class4"], axis=1)
y = npf["class2"]
c4_y = npf["class4"]

# used for the ridge feature selection (transform classes to -1 or 1)
y_f = classf[(y !="nonevent").astype(int)]

headers = list(X.columns.values)
feature_names = np.array(headers)

# -----------------------------------------------------------------------

# --- Sequential Feature selection using ridgeCV -------------------------------
# https://scikit-learn.org/stable/modules/feature_selection.html
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html

ridge = RidgeCV(alphas=np.logspace(-6,6, num=5))

sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=10, direction="forward"
)

sfs_forward.fit(X,y_f)
#print(sfs_forward.get_support())
best_features=feature_names[sfs_forward.get_support()]
print(F"Best 10 features: {best_features}")
print("\n")

X = X[best_features]
X_test = X_test[best_features]

# Normalize training- and test-datasets
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------------------


# ---- Model setup and testing ----------------------------------------------------

def StratifiedCV(_model, _X, _y, _splits):
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # Splits the data into equally distributed datasets (based on output variable)
    skf = StratifiedKFold(n_splits=_splits, shuffle=True, random_state=seed) 
    accuracies = []

    for train_index, test_index in skf.split(_X, _y): 
        X_train_fold = _X[train_index]
        X_test_fold = _X[test_index] 
        y_train_fold = _y[train_index]
        y_test_fold = _y[test_index] 
        
        _model.fit(X_train_fold, y_train_fold) 

        # Requires exact match of correct class, I think:
        y_test_pred = model.predict(X_test_fold)
        score = accuracy_score(y_test_fold, y_test_pred)
        accuracies.append(score)

        # More relaxed scoring method, dictated by the model:
        #accuracies.append(_model.score(X_test_fold, y_test_fold))

    return np.mean(accuracies) 

models_text = ["KNeighbours5", "Gaussian", "logisticRegression"]

kernel = 1 * RBF(5)
models = [KNeighborsClassifier(5),
          GaussianProcessClassifier(kernel=kernel, random_state=seed, n_jobs=4),
          LogisticRegression(random_state=seed, max_iter=10000, solver='saga', multi_class='ovr', penalty='elasticnet', l1_ratio=0.6)]

print("Class 2 models: ")

# Split to training/validation sets, stratified based on 'class4' 
X_train, X_valid, y_train, y_valid = train_test_split(
    # Any sense to stratify based on class4 vs class2?
    X, y, train_size=math.floor(len(npf.index)*0.6), random_state=seed, shuffle=True, stratify=y
)


for i, model in enumerate(models):
    model.fit(X_train, y_train)
    print(model.classes_)
    
    print(F"[{models_text[i]}]: ")
    #print(F"  Validation score: {round(model.score(X_valid,y_valid), 4)}")
    # above equals below
    y_pred = model.predict(X_valid)
    print(F"  Accuracy score (validation): {round(accuracy_score(y_valid, y_pred), 4)}")

    # balanced-accuracy gets equal value (maybe because the dataset has equal distribution 50% event, 50% non-event)
    #print(F"  Balanced accuracy score (validation): {round(balanced_accuracy_score(y_valid, y_pred), 4)}")

    cv_scores = cross_val_score(model, X, y, cv=10)
    print(F"  CV-Accuracy: {np.mean(cv_scores)}")

    # StratifiedKFold 
    # ensure that each fold has an equal distribution of classes
    print(F'  StratifiedKFold CV-Accuracy: {StratifiedCV(model, X, y, 10)}')

    y_pred_p = model.predict_proba(X_valid)
    #exp(-log_loss())
    print(F"  Perplexity {math.exp(log_loss(y_valid, y_pred_p))}")

    print("\n")


c4_models_text = ["Gaussian (OVR)", "logisticRegression (OVR)", "logisticRegression (multinomial)"]

c4_models = [
          # one_vs_one = do classification for each possible output_label pair (doesn't support predict_proba directly)
          # one_vs_rest = do one classifier for each output_label
          GaussianProcessClassifier(kernel=kernel, random_state=seed, n_jobs=4, multi_class = "one_vs_rest"),
          LogisticRegression(random_state=seed, solver='lbfgs', multi_class='ovr'),
          # inherently multiclass
          LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial')]

print("Class 4: ")

c4_X_train, c4_X_valid, c4_y_train, c4_y_valid = train_test_split(
    X, c4_y, train_size=math.floor(len(npf.index)*0.6), random_state=seed, shuffle=True, stratify=c4_y
)

for i, c4_model in enumerate(c4_models):
    c4_model.fit(c4_X_train, c4_y_train)
    
    print(F"[{c4_models_text[i]}]: ")
    print(c4_model.classes_)
    #print(F"  Validation score: {round(c4_model.score(c4_X_valid,c4_y_valid), 4)}")
    
    y_pred = model.predict(X_valid)
    print(F"  Accuracy score (validation): {round(accuracy_score(y_valid, y_pred), 4)}")

    # balanced-accuracy gets equal value (maybe because the dataset has equal distribution 50% event, 50% non-event)
    # print(F"  Balanced accuracy score (validation): {round(balanced_accuracy_score(y_valid, y_pred), 4)}")
    
    cv_scores = cross_validate(c4_model, X, c4_y, scoring='accuracy', cv=10)
    print(F"  CV_score: {np.mean(cv_scores['test_score'])}")

    # This value is similar to the Stratified CV, when using the exact match method
    #b_cv_scores = cross_validate(c4_model, X, c4_y, scoring='balanced_accuracy', cv=10)
    #print(F"  Balanced CV_score: {np.mean(b_cv_scores['test_score'])}")

    print(F'  StratifiedKFold CV-Accuracy: {StratifiedCV(c4_model, X, c4_y, 10)}')
    
    c4_y_pred_p = c4_model.predict_proba(c4_X_valid)
    #print(c4_y_pred_p.shape)
    print(F"  Perplexity {math.exp(log_loss(c4_y_valid, c4_y_pred_p))}")

    print("\n")

# ----------------------------------------------------------


# --- Write results to file ---------------------------------------

# Choose model
chosen_model = models[1] # Gaussian OVR
chosen_c4_model = c4_models[0] # Gaussian OVR

# Estimate using the existing model
binary_accuracy_estimate = StratifiedCV(chosen_model, X, y, 10)

# train on train+validation data to produce final model
# use this model to make predictions
chosen_model.fit(X, y)
chosen_c4_model.fit(X, c4_y)

c4_pred = chosen_c4_model.predict(X_test)
pred = chosen_model.predict_proba(X_test)
#pred2 = chosen_model.predict(X_test)

f = open("answers.csv", "w")
f.write(F"{str(binary_accuracy_estimate)}\n")
f.write("class4,p\n")
for i, x in enumerate(pred):
    f.write(F"{c4_pred[i]},{np.format_float_positional(pred[i][0])}\n")

f.close()

# ---------------------------------------------------------


