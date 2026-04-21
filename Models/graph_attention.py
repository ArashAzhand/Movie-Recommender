import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GATConv


class GAT(nn.Module):
    def __init__(self, in_user, in_movie, num_users, num_movies, hidden=64, heads=4):
        super().__init__()

        self.user_lin = nn.Linear(in_user, hidden)
        self.movie_lin = nn.Linear(in_movie, hidden)

        self.dropout = nn.Dropout(0)

        self.user_emb = nn.Embedding(num_users, hidden)
        self.movie_emb = nn.Embedding(num_movies, hidden)

        self.res_user = nn.Linear(hidden, hidden)
        self.res_movie = nn.Linear(hidden, hidden)


        # Layer 1
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'):
                GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False, dropout=0.2),

            ('movie', 'rated_by', 'user'):
                GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False, dropout=0.2),
        })

        # Layer 2
        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'):
                GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False, dropout=0.2),

            ('movie', 'rated_by', 'user'):
                GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False, dropout=0.2),
        })

    def forward(self, data):

        device = data['user'].x.device

        num_users = data['user'].num_nodes
        num_movies = data['movie'].num_nodes

        user_ids = torch.arange(num_users, device=device)
        movie_ids = torch.arange(num_movies, device=device)

        user_x = self.user_lin(data['user'].x) + self.user_emb(user_ids)
        movie_x = self.movie_lin(data['movie'].x) + self.movie_emb(movie_ids)

        x_dict = {
            'user': user_x,
            'movie': movie_x
        }

        # ---- Layer 1 ----
        x_dict = self.conv1(x_dict, data.edge_index_dict)

        # x_dict['user'] = self.res_user(user_x) + x_dict['user']
        # x_dict['movie'] = self.res_movie(movie_x) + x_dict['movie']

        x_dict = {
            key: self.dropout(torch.relu(x))
            for key, x in x_dict.items()
        }

        # ---- Layer 2 ----
        x_dict = self.conv2(x_dict, data.edge_index_dict)

        # x_dict['user'] = self.res_user(user_x) + x_dict['user']
        # x_dict['movie'] = self.res_movie(movie_x) + x_dict['movie']

        x_dict = {
            key: self.dropout(torch.relu(x))
            for key, x in x_dict.items()
        }

        user_x = x_dict['user']
        movie_x = x_dict['movie']

        edge_index = data['user', 'rates', 'movie'].edge_index

        user_emb = user_x[edge_index[0]]
        movie_emb = movie_x[edge_index[1]]

        # dot product interaction
        pred = (user_emb * movie_emb).sum(dim=1)

        # bound to rating range
        pred = 5 * torch.sigmoid(pred)

        return pred.squeeze()
