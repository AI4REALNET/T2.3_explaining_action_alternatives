import torch
import torch.nn as nn

def _construct_sequential(in_size, layer_sizes, out_size):
    if len(layer_sizes) == 0:
        return nn.Sequential(nn.Linear(in_size, out_size))
    
    layers = [nn.Linear(in_size, layer_sizes[0]), nn.ReLU()]
    for i in range(1, len(layer_sizes)):
        layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_sizes[-1], out_size))
    return nn.Sequential(*layers)

class ZeroEmbedder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, x, *args):
        return torch.zeros(x.shape[0],self.embedding_size,dtype=torch.float32)

class StateEmbedder(nn.Module):
    def __init__(self, state_size, belief_size, state_embedding_size, layer_sizes):
        super().__init__()
        self.trunk = _construct_sequential(state_size+belief_size, layer_sizes, state_embedding_size)

    def forward(self, s, b):
        x = torch.cat([s, b], 1)
        return self.trunk(x)

class ActionEmbedder(nn.Module):
    def __init__(self, N_actions, action_embedding_size, layer_sizes):
        super().__init__()
        self.trunk = _construct_sequential(N_actions * 2, layer_sizes, action_embedding_size)

    def forward(self, a):
        return self.trunk(a.flatten(1))

class OutcomePredictor(nn.Module):
    def __init__(self, state_embedder, action_embedder, state_embedding_size, action_embedding_size, N_actions, N_outcomes, layer_sizes):
        super().__init__()
        self.state_embedder = state_embedder
        self.action_embedder = action_embedder
        in_size = state_embedding_size + action_embedding_size
        out_size = N_outcomes * N_actions * 2

        self.trunk = _construct_sequential(in_size, layer_sizes, out_size)

    def forward(self, s, b, a):
        s_emb = self.state_embedder(s, b)
        a_emb = self.action_embedder(a)
        x = torch.cat([s_emb, a_emb], 1)
        preds = self.trunk(x).reshape((*a.shape, -1))

        pred_mask = a > 0.5
        pred_mask[:,:,1] = torch.logical_and(pred_mask[:,:,0], pred_mask[:,:,1])
        return torch.where(pred_mask[...,None].repeat_interleave(preds.shape[-1], -1), preds, torch.zeros_like(preds))