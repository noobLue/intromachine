# intromachine
Intro to machine learning term project


## Install requirements:

Run code:

```
pip install -r requirements.txt
```


## Generate results: 

Selects and prints the top 10 features by using ridgeCV with sequential feature selection

Prints some validation scores for binary (class) and multiclass (class4) classifiers

Classifiers (class 2):
- 5-NearestNeighbors
- GaussianProcessClassifier
- LogisticRegression (lasso + ridge)
	
Classifiers (class 4):
- GaussianProcessClassifier (one-vs-rest)
- LogisticRegression (one-vs-rest)
- LogisticRegression (multinomial)

Accuracy scores:
- Validation score is a stratified training / validation split
- Regular cross-validation
- Stratified cross-validation
- Perplexity

Writes the test data predictions and other stuff required for challenge submission file to 'answers.csv'

Run code:
```
python temp.py
```

Example output


## TODO

Currently classifiers for class2 and class4 are separate, possibly leading to inconsistencies like class2 is predicted to be *event*, but class4 is predicted to be *non-event*. Possible answers? First predict event vs non-event and then predict which event the event should be.