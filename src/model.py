# 1.Logistic Regression
# We can get the probility of each class
# Class are discrete variables.
# The Linear Regression will not worked because it is used for continous variables
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(
  C=1.0,
  max_iter=500,
  solver="lbfgs"
)
# 2.Random Forest
# Rely on the consequense of all trees and make decision together
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
  n_estimators=100, # the number of the trees
  max_depth=20,
  min_samples_leaf=2,
  random_state=SEED_CPU,
  n_jobs=-1
)
# Training two models using exactly the same training data
lr_model.fit(
  X_train_classical, # features
  y_train_classical # labels
)

rf_model.fit(
  X_train_classical,
  y_train_classical
)


print("Both models have been trained.")
