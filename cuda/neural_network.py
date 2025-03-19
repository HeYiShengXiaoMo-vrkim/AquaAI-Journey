from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import numpy as np


X = np.array([[0., 0.],
              [1., 1.]])
y = np.array([0, 1])
print(X.shape)
print(y.shape)

clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='relu',
                    hidden_layer_sizes=(5, 2), max_iter=2000, tol=1e-4, verbose=True)
clf.fit(X, y)

predicted_value = clf.predict([[2, 2],
                               [-1, -2]])
print(predicted_value)
predicted_proba = clf.predict_proba([[2., 2.],
                                     [-1., -2.]])
print(predicted_proba)
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])

print([coef.shape for coef in clf.intercepts_])
print([coef for coef in clf.intercepts_])
