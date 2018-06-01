# DATA558 - Machine Learning

Polished python code required for one of the assignments for DATA558 - Statistical Machine Learning For Data Scientists at University of Washington.

## Linear Support Vector Machine with Square Hinge Loss (Classification)

The model is implemented in [linear_svm.py]models/linear_svm.py. It uses fast gradient descent with backtracking and simplifies the usage to sklearn style .fit and .predict functions. Cross validation is used to find the optimal value of the regularization parameter.

## Demos

For testing with the spam dataset (real world) - Binary classifier
```
python3 demo_spam.py
```

For testing with the vowel dataset (real world) - Multinomial classifier using binary
```
python3 demo_vowel.py
```

## Usage

```python
from models import LinearSVM
LSVM = LinearSVM()
weights = LSVM.fit(train_features,train_labels)
test_predictions = LSVM.predict(weights,test_features)
```

# Required Libraries (Python 3)

numpy
sklearn
scipy
matplotlib
pandas
