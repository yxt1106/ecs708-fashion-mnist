# Predictions on the same validation set
# Compare the results of both models(seen in print out)
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  ConfusionMatrixDisplay
)
import pandas as pd

# Evaluate both models and get results
lr_pred = lr_model.predict(X_val_classical)
rf_pred = rf_model.predict(X_val_classical)


lr_accuracy = accuracy_score(
  y_val_classical,
  lr_pred
)

rf_accuracy = accuracy_score(
  y_val_classical,
  rf_pred
)


accuracy_comparison = pd.DataFrame({
  "Model": ["Logistic Regression", "Random Forest"],
  "Validation Accuracy": [
    lr_accuracy,
    rf_accuracy
  ]
})

display(accuracy_comparison)
# Evaluate performance of Logistic Regression and Random Forest on the validation set
# Logistic Regression report

# Get Report of Logistic Regression
lr_report = classification_report(
  y_val_classical,
  lr_pred,
  labels=np.arange(10),
  target_names=class_names,
  output_dict=True,
  zero_division=0
)

lr_table = pd.DataFrame(lr_report).T


# Random Forest report

# Get Report of Random Forest
rf_report = classification_report(
  y_val_classical,
  rf_pred,
  labels=np.arange(10),
  target_names=class_names,
  output_dict=True,
  zero_division=0
)

# Transfer results into DataFrame
rf_table = pd.DataFrame(rf_report).T


# Display class-level results

print("Logistic Regression")

display(
  lr_table.loc[
    class_names,
    ["precision", "recall", "f1-score", "support"]
  ]
)


print("Random Forest")

display(
  rf_table.loc[
    class_names,
    ["precision", "recall", "f1-score", "support"]
  ]
)
# Draw confusion matrix
# We can get how mach the model pridict is correct by check confusion matrix
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ConfusionMatrixDisplay.from_predictions(
  y_val_classical, # real labels
  lr_pred, # pridict labels
  display_labels=class_names,
  xticks_rotation=45,
  ax=axes[0],
  colorbar=False
)

axes[0].set_title("Logistic Regression")


ConfusionMatrixDisplay.from_predictions(
  y_val_classical,
  rf_pred,
  display_labels=class_names,
  xticks_rotation=45,
  ax=axes[1],
  colorbar=False
)

axes[1].set_title("Random Forest")

plt.tight_layout()
plt.show()
# Identify four types of predictions
# Compare result between Logitisc Regression and Rando Forest
groups = {
  "Both correct": np.where(
    (lr_pred == y_val_classical) &
    (rf_pred == y_val_classical)
  )[0],

  "LR wrong, RF correct": np.where(
    (lr_pred != y_val_classical) &
    (rf_pred == y_val_classical)
  )[0],

  "LR correct, RF wrong": np.where(
    (lr_pred == y_val_classical) &
    (rf_pred != y_val_classical)
  )[0],

  "Both wrong": np.where(
    (lr_pred != y_val_classical) &
    (rf_pred != y_val_classical)
  )[0]
}


# Pick two random examples from each group

rng = np.random.default_rng(42)

selected = []

for group_name, indices in groups.items():
  if len(indices) > 0:
    n = min(2, len(indices))
    chosen = rng.choice(
      indices,
      size=n,
      replace=False
    )
    for idx in chosen:
      selected.append((group_name, idx))


# Visualise the selected images

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

axes = axes.flatten()

for ax, (group_name, idx) in zip(axes, selected):
  image = X_val_classical[idx].reshape(28, 28)
  ax.imshow(image, cmap="gray")
  ax.set_title(
      f"{group_name}\n"
      f"True: {class_names[y_val_classical[idx]]}\n"
      f"LR: {class_names[lr_pred[idx]]}\n"
      f"RF: {class_names[rf_pred[idx]]}",
      fontsize=8
  )
  ax.axis("off")


for ax in axes[len(selected):]:
  ax.axis("off")


plt.tight_layout()
plt.show()
# The influence of C to model performance
C_values = [0.001, 0.01, 0.1, 1, 10]

lr_complexity_results = []

for c in C_values:
  model = LogisticRegression(
    C=c,
    max_iter=500,
    solver="lbfgs"
  )

  model.fit(
    X_train_classical,
    y_train_classical
  )

  train_accuracy = model.score(
    X_train_classical,
    y_train_classical
  )

  val_accuracy = model.score(
    X_val_classical,
    y_val_classical
  )

  lr_complexity_results.append({
    "C": c,
    "Training Accuracy": train_accuracy,
    "Validation Accuracy": val_accuracy
  })


lr_complexity_df = pd.DataFrame(lr_complexity_results)

display(lr_complexity_df)
# Research the performance of Random Forest

depth_values = [3, 5, 10, 20, None] # The test case of the deepth of tree

rf_complexity_results = []

for depth in depth_values:
  model = RandomForestClassifier(
    n_estimators=80, # number of trees
    max_depth=depth,
    min_samples_leaf=2,
    random_state=SEED_CPU,
    n_jobs=-1
  )

  model.fit(
    X_train_classical,
    y_train_classical
  )

  train_accuracy = model.score(
    X_train_classical,
    y_train_classical
  )

  val_accuracy = model.score(
    X_val_classical,
    y_val_classical
  )

  rf_complexity_results.append({
    "Max Depth": str(depth),
    "Training Accuracy": train_accuracy,
    "Validation Accuracy": val_accuracy
  })


rf_complexity_df = pd.DataFrame(rf_complexity_results)

display(rf_complexity_df)
# Generate figures
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


# Logistic Regression

axes[0].plot(
  lr_complexity_df["C"],
  lr_complexity_df["Training Accuracy"],
  marker="o",
  label="Training"
)

axes[0].plot(
  lr_complexity_df["C"],
  lr_complexity_df["Validation Accuracy"],
  marker="o",
  label="Validation"
)

axes[0].set_xscale("log")

axes[0].set_xlabel("C")
axes[0].set_ylabel("Accuracy")

axes[0].set_title("Logistic Regression Complexity")

axes[0].legend()
axes[0].grid(True)


# Random Forest

axes[1].plot(
  rf_complexity_df["Max Depth"],
  rf_complexity_df["Training Accuracy"],
  marker="o",
  label="Training"
)

axes[1].plot(
  rf_complexity_df["Max Depth"],
  rf_complexity_df["Validation Accuracy"],
  marker="o",
  label="Validation"
)

axes[1].set_xlabel("Maximum Tree Depth")
axes[1].set_ylabel("Accuracy")

axes[1].set_title("Random Forest Complexity")

axes[1].legend()
axes[1].grid(True)


plt.tight_layout()
plt.show()
# Choose the best C using validation accuracy
# By iterate with different value of C
best_C = lr_complexity_df.loc[
  lr_complexity_df["Validation Accuracy"].idxmax(),
  "C"
]

print("Best C:", best_C)


# Fit the selected Logistic Regression model

selected_model = LogisticRegression(
  C=best_C,
  max_iter=500,
  solver="lbfgs"
)

selected_model.fit(
  X_train_classical,
  y_train_classical
)


# Get probabilities for all 10 classes

probabilities = selected_model.predict_proba(
  X_val_classical
)


# Sum probabilities for footwear classes
footwear_labels = [5, 7, 9]

footwear_columns = [
  np.where(selected_model.classes_ == label)[0][0]
  for label in footwear_labels
]

p_footwear = probabilities[:, footwear_columns].sum(axis=1)


# Convert the actual labels into binary labels to get results
y_binary = np.isin(
  y_val_classical,
  footwear_labels
).astype(int)


print("Footwear probabilities:", p_footwear[:10])
print("Binary ground truth:", y_binary[:10])
# Thresholds optimization: Set an threholds to judge the class of item
# if > threholds , it's correct, otherwise it's incorrect
thresholds = np.linspace(0.05, 0.95, 19)

FP_COST = 1
FN_COST = 3

threshold_results = []

for threshold in thresholds:
  y_pred_binary = (
    p_footwear >= threshold
  ).astype(int)

  tn, fp, fn, tp = confusion_matrix(
    y_binary,
    y_pred_binary,
    labels=[0, 1]
  ).ravel()

  # calculate the probilities
  fpr = fp / (fp + tn)
  fnr = fn / (fn + tp)
  total_cost = (
    FP_COST * fp +
    FN_COST * fn
  )
  mean_cost = total_cost / len(y_binary)
  threshold_results.append({
    "Threshold": threshold,
    "FPR": fpr,
    "FNR": fnr,
    "FP": fp,
    "FN": fn,
    "Average Cost": mean_cost
  })


threshold_df = pd.DataFrame(threshold_results)
display(threshold_df)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))


# Error rates
axes[0].plot(
  threshold_df["Threshold"],
  threshold_df["FPR"],
  marker="o",
  label="False Positive Rate"
)

axes[0].plot(
  threshold_df["Threshold"],
  threshold_df["FNR"],
  marker="o",
  label="False Negative Rate"
)

axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Error Rate")

axes[0].set_title("Threshold vs Error Rates")

axes[0].legend()
axes[0].grid(True)


# Average cost

axes[1].plot(
  threshold_df["Threshold"],
  threshold_df["Average Cost"],
  marker="o"
)

axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Average Cost")

axes[1].set_title("Threshold vs Average Cost")

axes[1].grid(True)


plt.tight_layout()
plt.show()


# Find minimum cost threshold

best_threshold = threshold_df.loc[
  threshold_df["Average Cost"].idxmin()
]

print("Best threshold:")
display(best_threshold)
def loader_to_numpy(loader): # Transfer data into numpy format
  X, y = [], []

  for images, labels in loader:
    X.append(
      images.flatten(start_dim=1).numpy()
    )
    y.append(labels.numpy())

  return ( # concatenate batches
    np.concatenate(X), # image features
    np.concatenate(y) # image labels
  )


X_test_classical, y_test_classical = loader_to_numpy(
  test_loader
)


# Make probability columns follow class labels 0-9
def predict_proba_10(model, X):
  # The probability of each sample belonging to each category
  probabilities = model.predict_proba(X)
  result = np.zeros( # Create a fixed 10-column matrix
    (len(X), 10)
  )
  # Place the raw predicted probabilities into the positions corresponding to their respective classes.
  result[:, model.classes_.astype(int)] = probabilities
  return result


# Validation probabilities
# Predicted probabilities from the LR model on the validation set
val_lr_prob = predict_proba_10(
  lr_model,
  X_val_classical
)

# Predicted probabilities from the RF model on the validation set
val_rf_prob = predict_proba_10(
  rf_model,
  X_val_classical
)


# Test probabilities
test_lr_prob = predict_proba_10(
  lr_model,
  X_test_classical
)

test_rf_prob = predict_proba_10(
  rf_model,
  X_test_classical
)
from sklearn.metrics import accuracy_score

weights = [0.25, 0.5, 0.75]
weight_results = []

for alpha in weights:
  ensemble_prob = ( # Combining probabilities from LR and RF on the validation set
    alpha * val_lr_prob
    + (1 - alpha) * val_rf_prob
  )
  pred = ensemble_prob.argmax(axis=1) # Select the category with the highest probability for each image.
  acc = accuracy_score(
    y_val_classical,
    pred
  )
  weight_results.append({
    "LR Weight": alpha,
    "RF Weight": 1 - alpha,
    "Validation Accuracy": acc
  })


weight_df = pd.DataFrame(weight_results)
display(weight_df)


# Select the best weight using validation accuracy
best_alpha = weight_df.loc[
  weight_df["Validation Accuracy"].idxmax(),
  "LR Weight"
]

print("Selected LR weight:", best_alpha)
print("Selected RF weight:", 1 - best_alpha)
# Ensemble probabilities
# Combine the predicted probabilities from two models with the best weights
val_ensemble_prob = (
  best_alpha * val_lr_prob
  + (1 - best_alpha) * val_rf_prob
)

test_ensemble_prob = (
  best_alpha * test_lr_prob
  + (1 - best_alpha) * test_rf_prob
)

# Organize data
models_val = {
  "Logistic Regression": val_lr_prob,
  "Random Forest": val_rf_prob,
  "Ensemble": val_ensemble_prob
}

models_test = {
  "Logistic Regression": test_lr_prob,
  "Random Forest": test_rf_prob,
  "Ensemble": test_ensemble_prob
}


def overall_results(models, y_true):
  rows = []

  for name, probabilities in models.items():
    predictions = probabilities.argmax(axis=1)
    rows.append({
      "Model": name,
      "Accuracy": accuracy_score(
        y_true,
        predictions
      )
    })

  return pd.DataFrame(rows)


print("Validation Results")

display(
  overall_results(
    models_val,
    y_val_classical
  )
)


print("Test Results")

display(
  overall_results(
    models_test,
    y_test_classical
  )
)
# Show Datas
from sklearn.metrics import classification_report

class_results = []

for name, probabilities in models_test.items():
  predictions = probabilities.argmax(axis=1)
  report = classification_report(
    y_test_classical,
    predictions,
    labels=np.arange(10),
    target_names=class_names,
    output_dict=True,
    zero_division=0
  )

  for class_name in class_names:
    class_results.append({
      "Model": name,
      "Class": class_name,
      "Precision": report[class_name]["precision"],
      "Recall": report[class_name]["recall"],
      "F1": report[class_name]["f1-score"],
      "Support": report[class_name]["support"]
    })


class_results_df = pd.DataFrame(class_results)
display(class_results_df)
