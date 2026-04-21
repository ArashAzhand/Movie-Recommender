import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


# Local imports
from Utils.data_utils import load_ml_100k_data, preprocess_users, preprocess_items, load_ml_100k_fold, create_heterodata
from Models.graph_sage import GraphSAGE  
from Models.light_gcn import LightGCN
from Models.graph_attention import GAT


# ===========================================================
# Configuration
# ===========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Data/ml-100k/"

EPOCHS = 20
LR = 1e-3


# ===========================================================
# Load and preprocess data
# ===========================================================
print("\n--- Loading and Preprocessing Data ---")
users_df, items_df, _ = load_ml_100k_data(DATA_DIR)
proc_users_df = preprocess_users(users_df)
proc_items_df = preprocess_items(items_df)





# ===========================================================
# Evaluation
# ===========================================================
def evaluate(model, loader, device):
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            pred = model(data)
            true = data['user','rates','movie'].edge_attr.to(device).squeeze()

            preds.append(pred.cpu())
            trues.append(true.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)

    return rmse, r2



# ===========================================================
# Training loop
# ===========================================================
all_rmse = []
all_r2 = []
max_rmse = np.inf
criterion = nn.MSELoss()

for fold in range(1, 6):

    print(f"\n============================")
    print(f"Training Fold {fold}")
    print(f"============================")

    train_df, test_df = load_ml_100k_fold(DATA_DIR, fold_idx=fold)

    hetero_train = create_heterodata(proc_users_df, proc_items_df, train_df)
    hetero_test  = create_heterodata(proc_users_df, proc_items_df, test_df)

    train_loader = DataLoader([hetero_train], batch_size=1)
    test_loader = DataLoader([hetero_test], batch_size=1)

    # ===============================
    # Select Model
    # ===============================
    print("\n--- Initializing Model ---")
    MODEL_SAVE_PATH = "Trained_Models/best_GAT.pt"

    # ------GraphSAGE------
    # model = GraphSAGE(
    #     in_user=hetero_train['user'].x.shape[1],
    #     in_movie=hetero_train['movie'].x.shape[1],
    #     num_users=hetero_train['user'].num_nodes,
    #     num_movies=hetero_train['movie'].num_nodes,
    #     hidden=64
    # ).to(DEVICE)


    # ------LightGCN------
    # model = LightGCN(
    #     # in_user=hetero_train['user'].x.shape[1],
    #     # in_movie=hetero_train['movie'].x.shape[1],
    #     num_users=hetero_train['user'].num_nodes,
    #     num_movies=hetero_train['movie'].num_nodes,
    #     hidden=64,
    #     num_layers=3
    # ).to(DEVICE)


    # ------GAT------
    model = GAT(
        in_user=hetero_train['user'].x.shape[1],
        in_movie=hetero_train['movie'].x.shape[1],
        num_users=hetero_train['user'].num_nodes,
        num_movies=hetero_train['movie'].num_nodes,
        hidden=64,
        heads=8
    ).to(DEVICE)


    # ===============================



    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)



    # ---- training ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()

            pred_ratings = model(data)
            # print(f"min:{pred_ratings.min():.3f}, max:{pred_ratings.max():.3f}, mean:{pred_ratings.mean():.3f}")

            true_ratings = data['user', 'rates', 'movie'].edge_attr.to(DEVICE).squeeze()

            loss = criterion(pred_ratings, true_ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Fold {fold} Epoch {epoch} Loss: {avg_loss:.4f}")

    
    # ---- evaluation ----
    rmse, r2 = evaluate(model, test_loader, DEVICE)

    if rmse < max_rmse:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model with lowest RMSE: {rmse:.4f} saved at: {MODEL_SAVE_PATH}")
        max_rmse = rmse

    print(f"\nFold {fold} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    all_rmse.append(rmse)
    all_r2.append(r2)



print("\n============================")
print("5-Fold Cross Validation Results")
print("============================")

print(f"RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
print(f"R2  : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")

