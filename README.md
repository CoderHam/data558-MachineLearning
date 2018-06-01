# DATA558 - Machine Learning

Polished python code required for one of the assignments for DATA558 - Statistical Machine Learning For Data Scientists at University of Washington.

## Linear Support Vector Machine with Square Hinge Loss (Classification)

The model is implemented in [linear_svm.py]models/linear_svm.py. It uses fast gradient descent with backtracking and simplifies the usage to sklearn style .fit and .predict functions. Cross validation is used to find the optimal value of the regularization parameter.

## Demos

For testing with the Spam dataset (from the book The Elements of Statistical Learning) - Binary classifier
```
python3 demo_spam.py
```

For testing with the Vowel dataset (from the book The Elements of Statistical Learning) - Multinomial classifier is built using binary classifiers in one-vs-one style.
```
python3 demo_vowel.py
```

For testing with a custom generated dataset (simulated) - Binary classifier. Bonus - compare performance with sklearn
```
python3 demo_simulated.py
```

For comparing custom implemented with sklearn on spam dataset (real world) - Binary classifier
```
python3 compare_spam.py
```

## Usage

```python
from models import LinearSVM
LSVM = LinearSVM()
weights = LSVM.fit(train_features,train_labels)
test_predictions = LSVM.predict(weights,test_features)
```


## Data

The data is present in the __data__ folder and can also be downloaded from https://web.stanford.edu/~hastie/ElemStatLearn/data.html. There are a few other datasets available there to play around with.

# Required Libraries (Python 3)

numpy
sklearn
scipy
matplotlib
pandas
