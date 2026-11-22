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
