import torch
from torch import nn
from torch_geometric.nn import LGConv


class LightGCN(nn.Module):
    def __init__(self, in_user, in_movie, num_users, num_movies, hidden=64, num_layers=2):
        super().__init__()

        self.num_layers = num_layers

        # feature projection
        self.user_lin = nn.Linear(in_user, hidden)
        self.movie_lin = nn.Linear(in_movie, hidden)

        # embeddings
        self.user_emb = nn.Embedding(num_users, hidden)
        self.movie_emb = nn.Embedding(num_movies, hidden)

        # LightGCN layers
        self.convs = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, data):

        device = data['user'].x.device

        num_users = data['user'].num_nodes
        num_movies = data['movie'].num_nodes

        user_ids = torch.arange(num_users, device=device)
        movie_ids = torch.arange(num_movies, device=device)

        # initial embeddings
        user_x = self.user_emb(user_ids) # self.user_lin(data['user'].x) + self.user_emb(user_ids)
        movie_x = self.movie_emb(movie_ids) # self.movie_lin(data['movie'].x) + self.movie_emb(movie_ids)

        x = torch.cat([user_x, movie_x], dim=0)

        # build bipartite edge index
        edge_index = data['user','rates','movie'].edge_index

        user_offset = 0
        movie_offset = num_users

        edge_user = edge_index[0]
        edge_movie = edge_index[1] + movie_offset

        edge_index = torch.stack([
            torch.cat([edge_user, edge_movie]),
            torch.cat([edge_movie, edge_user])
        ])

        # LightGCN propagation
        embs = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)

        # average embeddings
        x = torch.stack(embs, dim=0).mean(dim=0)

        user_x = x[:num_users]
        movie_x = x[num_users:]

        # prediction edges
        edge_index = data['user','rates','movie'].edge_index

        user_emb = user_x[edge_index[0]]
        movie_emb = movie_x[edge_index[1]]

        # pred = (user_emb * movie_emb).sum(dim=1)

        # pred = 5 * torch.sigmoid(pred)
        x = torch.cat([user_emb, movie_emb], dim=1)
        pred = self.regressor(x).squeeze()
        pred = torch.clamp(pred, 0, 5)


        return pred.squeeze()
