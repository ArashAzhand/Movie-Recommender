#  Graph Neural Networks for Rating Prediction in Recommender Systems

A graph-based recommender system built with **PyTorch Geometric** to predict user ratings on movies using the **MovieLens 100K** dataset.  
This project explores how **Graph Neural Networks (GNNs)** such as **GraphSAGE**, **LightGCN**, and **GAT** can model user-item interactions more effectively than traditional methods.

---

##  Project Overview

Modern platforms like Netflix, Amazon, and Spotify rely heavily on recommender systems to personalize content.

Instead of using traditional matrix factorization only, this project models recommendation data as a **heterogeneous bipartite graph**:

-  Users = one node type  
-  Movies = another node type  
-  Ratings = edges between users and movies  

The goal is to predict unseen ratings:

\[
f(u,i) \rightarrow \hat{r}_{ui}
\]

Where:

- \(u\) = user  
- \(i\) = movie  
- \(\hat{r}_{ui}\) = predicted rating  

---

##  Rating Prediction Formula

After learning user and movie embeddings, the predicted rating is computed using their interaction:

\[
\hat{r}_{ui} = 5 \cdot \sigma(e_u^T e_i)
\]

Where:

- \(e_u\) = learned embedding of user \(u\)  
- \(e_i\) = learned embedding of movie \(i\)  
- \(e_u^T e_i\) = dot product similarity  
- \(\sigma(x)\) = sigmoid activation  

This bounds predictions into the valid MovieLens rating range:

\[
1 \leq \hat{r}_{ui} \leq 5
\]

---

##  Dataset

### MovieLens 100K

- 100,000 ratings
- 943 users
- 1,682 movies
- Ratings from **1 to 5**

Each record contains:

- User ID  
- Movie ID  
- Rating  
- Timestamp  

---

##  Graph Representation

The dataset is converted into a `HeteroData` graph using PyTorch Geometric.

### Node Types

- `user`
- `movie`

### Edge Types

- `user -> rates -> movie`
- `movie -> rev_rates -> user`

Reverse edges are added for bidirectional message passing.

---

##  Models Implemented

###  GraphSAGE
Neighborhood aggregation using mean pooling.

###  LightGCN
Simplified graph convolution for recommendation.

###  Graph Attention Networks (GAT)
Uses attention scores to weigh neighbors differently.

---

##  Training Objective

The model minimizes **Mean Squared Error (MSE)**:

\[
MSE = \frac{1}{N}\sum(r_{ui}-\hat{r}_{ui})^2
\]

Evaluation metric:

\[
RMSE = \sqrt{\frac{1}{N}\sum(r_{ui}-\hat{r}_{ui})^2}
\]

---

##  Results (5-Fold Cross Validation)

| Model | RMSE | R² |
|------|------|------|
| 1-layer GraphSAGE | 1.0927 | 0.0574 |
| 2-layer GraphSAGE + Dropout | 1.1010 | 0.0428 |
| ⭐ 2-layer GraphSAGE + Embeddings | **1.0802** | **0.0785** |
| 2-layer LightGCN + Embeddings | 1.5531 | -0.9061 |
| 2-layer GAT + Embeddings | 1.1431 | -0.0316 |

---

##  Key Findings

✅ Adding learnable **user/movie embeddings** significantly improved performance.  
✅ GraphSAGE was the most stable and accurate architecture.  
✅ LightGCN underperformed because it is better suited for ranking tasks rather than explicit rating prediction.  
✅ Deeper models sometimes suffered from oversmoothing.  
✅ Dropout was not beneficial on this small dataset.



