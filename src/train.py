transform = transforms.ToTensor()

# Get Train & Test data
train_data = datasets.FashionMNIST(
  root="./data",
  train=True,
  download=True,
  transform=transform
)

test_data = datasets.FashionMNIST(
  root="./data",
  train=False,
  download=True,
  transform=transform
)

train_labels = train_data.targets.numpy()
test_labels = test_data.targets.numpy()

print("Train:", len(train_labels))
print("Test:", len(test_labels))
# Random seeds
# Random seed: Ensures that data is randomly assigned,
# while also guaranteeing that, due to this random assignment,
# no data points end up in the wrong dataset.
SEED_SPLIT = 42
SEED_CPU = 43
SEED_CLUSTER = 44

# Store the index of training dataset
train_all_indices = np.arange(len(train_data))

# Split the official training set, stratified and seeded: 50,000 train / 10,000 validation
train_indices, val_indices = train_test_split(
  train_all_indices,
  test_size=10000,
  random_state=SEED_SPLIT,
  stratify=train_labels
)

print("Train:", len(train_indices))
print("Validation:", len(val_indices))
# Helper function
# Extract specific number of samples from existing data index with a certain proportion
def stratified_sample(base_indices, labels, n_samples, seed):
  local_indices = np.arange(len(base_indices))
  selected_local, _ = train_test_split(
    local_indices,
    train_size=n_samples,
    random_state=seed,
    stratify=labels[base_indices]
  )
  # The indices relative to original dataset
  return base_indices[selected_local]

# Reduced dataset from 5,0000 to 10,000
train_indices = stratified_sample(
  train_indices,
  train_labels,
  n_samples=10000,
  seed=SEED_CPU
)

# Clustering / feature extraction subsets with stratified_sample

# Extract 3000 data from training set
cluster_train_indices = stratified_sample(
  train_indices,
  train_labels,
  n_samples=3000,
  seed=SEED_CLUSTER
)

# Same as above, but this is validation datasets
cluster_val_indices = stratified_sample(
  val_indices,
  train_labels,
  n_samples=1000,
  seed=SEED_CLUSTER
)

test_all_indices = np.arange(len(test_data))

cluster_test_indices = stratified_sample(
  test_all_indices,
  test_labels,
  n_samples=1000,
  seed=SEED_CLUSTER
)

# Create subset from dataset according to specific index
# e.g. train_indices
train_dataset = Subset(
  train_data, train_indices.tolist()
)

val_dataset = Subset(
  train_data, val_indices.tolist()
)

test_dataset = test_data

cluster_train_dataset = Subset(
  train_data, cluster_train_indices.tolist()
)

cluster_val_dataset = Subset(
  train_data, cluster_val_indices.tolist()
)

cluster_test_dataset = Subset(
  test_data, cluster_test_indices.tolist()
)

# Create DataLoaders
# Read the data from dataset to model for training or validation
train_loader = DataLoader(
  train_dataset,
  batch_size=64,
  shuffle=True,
  num_workers=0
)

# CPU-based training option
train_loader = DataLoader(
  train_dataset,
  batch_size=64,
  shuffle=True,
  num_workers=0
)

# Validation and test
val_loader = DataLoader(
  val_dataset,
  batch_size=128,
  shuffle=False,
  num_workers=0
)

test_loader = DataLoader(
  test_dataset,
  batch_size=128,
  shuffle=False,
  num_workers=0
)

# Clustering / feature extraction
cluster_train_loader = DataLoader(
  cluster_train_dataset,
  batch_size=128,
  shuffle=False,
  num_workers=0
)

cluster_val_loader = DataLoader(
  cluster_val_dataset,
  batch_size=128,
  shuffle=False,
  num_workers=0
)

cluster_test_loader = DataLoader(
  cluster_test_dataset,
  batch_size=128,
  shuffle=False,
  num_workers=0
)

np.savez_compressed(
  "fashion_mnist_indices.npz",

  seed_split=SEED_SPLIT,
  seed_cpu=SEED_CPU,
  seed_cluster=SEED_CLUSTER,

  train_indices=train_indices,
  val_indices=val_indices,

  cpu_train_indices=train_indices,

  cluster_train_indices=cluster_train_indices,
  cluster_val_indices=cluster_val_indices,
  cluster_test_indices=cluster_test_indices
)

# The data length we get
print("Main Training:", len(train_dataset))
print("CPU Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

print("Cluster Training:", len(cluster_train_dataset))
print("Cluster Validation:", len(cluster_val_dataset))
print("Cluster Test:", len(cluster_test_dataset))
# Class names
class_names = train_data.classes
class_names

# Linear Regression
from sklearn.linear_model import LogisticRegression
def loader_to_numpy(loader):
  images_list = []
  labels_list = []

  for images, labels in loader:
    # [batch_size, 1, 28, 28] -> [batch_size, 784]
    images = images.flatten(start_dim=1)
    images_list.append(images.numpy())
    labels_list.append(labels.numpy())

  X = np.concatenate(images_list).astype(np.float32)
  y = np.concatenate(labels_list).astype(np.int64)
  return X, y


# Training subset
classical_train_loader = DataLoader(
  train_dataset,
  batch_size=256,
  shuffle=False,
  num_workers=0
)

# Validation set
classical_val_loader = DataLoader(
  val_dataset,
  batch_size=256,
  shuffle=False,
  num_workers=0
)


# Extract data
X_train_classical, y_train_classical = loader_to_numpy(
  classical_train_loader
)

X_val_classical, y_val_classical = loader_to_numpy(
  classical_val_loader
)


print("Classical training:", X_train_classical.shape)
print("Validation:", X_val_classical.shape)
