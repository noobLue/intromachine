# intromachine
Intro to machine learning term project

*Note: Some of this readme is outdated as of 12th of December*


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
```
\intromachine>python temp.py
Best 10 features: ['CO242.mean' 'H2O672.mean' 'NO42.std' 'NOx672.std' 'O384.mean'
 'RHIRGA168.std' 'RHIRGA42.mean' 'T168.std' 'CS.mean' 'CS.std']


Class 2 models:
['event' 'nonevent']
[KNeighbours5]:
  Accuracy score (validation): 0.8441
  CV-Accuracy: 0.8646160962072156
  StratifiedKFold CV-Accuracy: 0.8768732654949121
  Perplexity 1.207708940534902


['event' 'nonevent']
[Gaussian]:
  Accuracy score (validation): 0.871
  CV-Accuracy: 0.8860314523589269
  StratifiedKFold CV-Accuracy: 0.8878353376503236
  Perplexity 1.2680434185203733


['event' 'nonevent']
[logisticRegression]:
  Accuracy score (validation): 0.8763
  CV-Accuracy: 0.8925531914893616
  StratifiedKFold CV-Accuracy: 0.8900092506938021
  Perplexity 1.2706654518204434


Class 4:
[Gaussian (OVR)]:
['II' 'Ia' 'Ib' 'nonevent']
  Accuracy score (validation): 0.8978
  CV_score: 0.6552266419981498
  StratifiedKFold CV-Accuracy: 0.4570767807585569
  Perplexity 1.9670481655274807


[logisticRegression (OVR)]:
['II' 'Ia' 'Ib' 'nonevent']
  Accuracy score (validation): 0.8978
  CV_score: 0.6469010175763182
  StratifiedKFold CV-Accuracy: 0.4570767807585569
  Perplexity 2.0535890539568857


[logisticRegression (multinomial)]:
['II' 'Ia' 'Ib' 'nonevent']
  Accuracy score (validation): 0.8978
  CV_score: 0.6382516188714153
  StratifiedKFold CV-Accuracy: 0.4570767807585569
  Perplexity 2.049037353324779
```

## TODO

Currently classifiers for class2 and class4 are separate, possibly leading to inconsistencies like class2 is predicted to be *event*, but class4 is predicted to be *non-event*. Possible answers? First predict event vs non-event and then predict which event the event should be.

- Answered by prioritizing the binary classifier. If the binary classifier chooses "event" then for the multiclass-classifier we pick the event with the highest probability (non-event excluded). If the binary classifier says "nonevent" then we respond "nonevent" for the multiclass question as well. 