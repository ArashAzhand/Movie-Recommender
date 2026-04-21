
# Graph Neural Networks for Rating Prediction

This project explores the use of **Graph Neural Networks (GNNs)** for predicting user ratings in recommender systems, using the **MovieLens 100K dataset**.

The goal is to model user–item interactions as a graph and learn meaningful representations that improve rating prediction accuracy compared to traditional methods like matrix factorization.

---

## Overview

Modern recommender systems (e.g., Netflix, Spotify, Amazon) rely on predicting user preferences. In this project, we:

* Formulate recommendation as a **regression problem**
* Represent the dataset as a **bipartite graph**
* Apply **Graph Neural Networks** to learn user and item embeddings
* Compare multiple GNN architectures

---

## Dataset

We use the **MovieLens 100K** dataset:

*  100,000 ratings
*  943 users
*  1682 movies
*  Ratings from 1 to 5

Each interaction is treated as an edge between a user and a movie.

---

##  Problem Formulation

We aim to learn a function:

```
f(u, i) → r̂_ui
```

Where:

* `u` = user
* `i` = movie
* `r̂_ui` = predicted rating

This is treated as a **regression task**.

---

##  Graph Representation

* Nodes:

  * Users
  * Movies
* Edges:

  * User → Movie (rating interaction)
  * Movie → User (reverse edges for message passing)

Implemented using **PyTorch Geometric (HeteroData)**.

---

##  Models Implemented

We experimented with several GNN architectures:

* **GraphSAGE**
* **LightGCN**
* **Graph Attention Networks (GAT)**

---

## Training

* **Loss Function:** Mean Squared Error (MSE)
* **Evaluation Metric:** Root Mean Squared Error (RMSE)
* **Cross Validation:** 5-fold

Prediction is computed using a **dot product** between user and movie embeddings.

---

##  Key Techniques

* Learnable **user & movie embeddings**
* Feature normalization for stable training
* Bounding predictions using sigmoid scaling
* Bidirectional message passing
* Dot-product predictor instead of MLP

---

##  Results

| Model                            | RMSE       | R²         |
| -------------------------------- | ---------- | ---------- |
| GraphSAGE (2-layer + embeddings) | **1.0802** | **0.0785** |
| GraphSAGE (1-layer)              | 1.0927     | 0.0574     |
| GAT (best variant)               | 1.1242     | 0.0023     |
| LightGCN                         | 1.5531     | -0.9061    |

 **Best Model:**
 2-layer GraphSAGE with embeddings

---

##  Key Observations

* Adding embeddings significantly improves performance
* Deeper models without embeddings suffer from **oversmoothing**
* Dropout negatively impacted performance on this small dataset
* LightGCN underperformed since it is better suited for ranking tasks
* GAT models are sensitive to missing self-loops


