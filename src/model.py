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
class MLP(nn.Module):
  def __init__(self, hidden_sizes=(256, 128), num_classes=10):
    super().__init__()
    self.hidden_layers = nn.ModuleList()
    input_size = 784

    # Hidden layers
    for hidden_size in hidden_sizes:
      layer = nn.Linear(
        input_size,
        hidden_size
      )
      self.hidden_layers.append(layer)
      input_size = hidden_size

    # Output layer
    self.output_layer = nn.Linear(
      input_size,
      num_classes
    )

    # Weight initialization
    for layer in self.hidden_layers:
      nn.init.kaiming_uniform_(
        layer.weight,
        nonlinearity="relu"
      )
      nn.init.zeros_(layer.bias)

    # Output layer initialization
    nn.init.xavier_uniform_(
      self.output_layer.weight
    )
    nn.init.zeros_(
      self.output_layer.bias
    )

  def forward(self, x):
    # Flatten [batch, 1, 28, 28] -> [batch, 784]
    x = torch.flatten(x, start_dim=1)
    # Hidden layers
    for layer in self.hidden_layers:
      x = F.relu(layer(x))
    # Output logits
    x = self.output_layer(x)
    return x
