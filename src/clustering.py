# Part 4 — Clustering: structure without labels
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

from scipy.cluster.hierarchy import linkage, dendrogram

# Fixed random subset, without using labels

# No labels are used during training here;
# the model discovers similarities and structures within the data on its own rather than label
rng = np.random.default_rng(44)

# Select 3000 images to reduce calculation
cluster_indices = np.sort(
    rng.choice(train_indices, 3000, replace=False)
)

# Flatten image because K-Means cannot be applied to four-dimensional image.
images = torch.stack([
    train_data[i][0] for i in cluster_indices
])

X_cluster = images.flatten(start_dim=1).numpy()

# Keep labels separately for later interpretation only
y_cluster = train_labels[cluster_indices]

images_cluster = images.squeeze(1).numpy()

print(X_cluster.shape)
K_values = [2, 5, 8, 10, 12, 15, 20] # k clusters
k_results = []

# Fixed sample for Silhouette
sil_idx = np.random.default_rng(123).choice(
  len(X_cluster),
  size=1000,
  replace=False
)

for k in K_values:
    model = KMeans(
      n_clusters=k,
      init="k-means++",
      n_init=10,
      max_iter=300,
      random_state=44
    )

    labels = model.fit_predict(X_cluster)
    score = silhouette_score(
        X_cluster[sil_idx],
        labels[sil_idx]
    )
    k_results.append({
      "K": k,
      "Inertia": model.inertia_, # The sum of squared distances within clusters
      # Intra-cluster compactness and inter-cluster separation
      "Silhouette": score
    })

k_df = pd.DataFrame(k_results)
display(k_df)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(k_df["K"], k_df["Inertia"], marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")

axes[1].plot(k_df["K"], k_df["Silhouette"], marker="o")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.show()
chosen_k = int(
  k_df.loc[k_df["Silhouette"].idxmax(), "K"]
)

seeds = [11, 22, 33]

stability = {}

for n_init in [1, 10]:
  for seed in seeds:
    model = KMeans(
      n_clusters=chosen_k,
      init="k-means++",
      n_init=n_init,
      max_iter=300,
      random_state=seed
    )

    labels = model.fit_predict(X_cluster)

    stability[(n_init, seed)] = {
      "labels": labels,
      "inertia": model.inertia_
    }

# Pairwise ARI
rows = []

for n_init in [1, 10]:
  aris = []
  for i in range(len(seeds)):
    for j in range(i + 1, len(seeds)):
        a = stability[(n_init, seeds[i])]["labels"]
        b = stability[(n_init, seeds[j])]["labels"]
        aris.append(adjusted_rand_score(a, b))

  rows.append({
    "n_init": n_init,
    "Mean Inertia": np.mean([
      stability[(n_init, s)]["inertia"]
      for s in seeds
    ]),
    "Mean ARI": np.mean(aris)
  })

display(pd.DataFrame(rows))
final_model = KMeans(
  n_clusters=chosen_k,
  init="k-means++",
  n_init=10,
  max_iter=300,
  random_state=44
)

cluster_labels = final_model.fit_predict(X_cluster)

print("Selected K:", chosen_k)
print("Inertia:", final_model.inertia_)
class_names = train_data.classes

composition = pd.crosstab(
  cluster_labels,
  y_cluster
).reindex(columns=range(10), fill_value=0)

composition.columns = class_names

display(composition)

display(
  composition.div(
    composition.sum(axis=1),
    axis=0
  ).round(3)
)
fig, axes = plt.subplots(
  chosen_k, 3,
  figsize=(9, chosen_k * 1.5),
  squeeze=False
)

rng = np.random.default_rng(123)

for k in range(chosen_k):
  members = np.where(cluster_labels == k)[0]
  selected = rng.choice(
    members,
    size=min(3, len(members)),
    replace=False
  )

  for j, idx in enumerate(selected):
    axes[k, j].imshow(
      images_cluster[idx],
      cmap="gray"
    )

    axes[k, j].set_title(
      class_names[y_cluster[idx]],
      fontsize=8
    )

    axes[k, j].axis("off")

  for j in range(len(selected), 3):
    axes[k, j].axis("off")

plt.tight_layout()
plt.show()
# Fixed subset of 400 images
rng = np.random.default_rng(404)

idx = np.sort(
  rng.choice(len(X_cluster), 400, replace=False)
)

X_hier = X_cluster[idx]

y_hier = y_cluster[idx]

# Ward linkage
Z = linkage(
  X_hier,
  method="ward",
  metric="euclidean"
)

# Dendrogram
plt.figure(figsize=(12, 5))

dendrogram(
  Z,
  truncate_mode="lastp",
  p=25
)

plt.title("Hierarchical Clustering - Ward Linkage")
plt.xlabel("Cluster")
plt.ylabel("Merge Distance")

plt.tight_layout()
plt.show()
hier_labels = AgglomerativeClustering(
  n_clusters=chosen_k,
  linkage="ward"
).fit_predict(X_hier)

# Compare with K-means on the same 400 images
kmeans_subset = KMeans(
  n_clusters=chosen_k,
  n_init=10,
  random_state=44
).fit_predict(X_hier)

print(
  "K-means vs Hierarchical ARI:",
  adjusted_rand_score(kmeans_subset, hier_labels)
)
