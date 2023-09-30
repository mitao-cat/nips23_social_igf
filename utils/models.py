import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):    # (N,) (N,)
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class BaseMF(nn.Module):
    def __init__(self, args):
        super(BaseMF, self).__init__()
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.device = args.device
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_size)
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_size)
        self.bprloss = BPRLoss()
        self.apply(xavier_normal_initialization)

    def predict(self):
        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight # (self.num_item, dim)
        output = user_embedding @ item_embedding.transpose(-1, -2)
        eval_output = output.detach()
        return eval_output

    def calculate_pos_neg_scores(self, users, pos_items, neg_items):
        user_e = self.user_embedding(users)
        pos_e, neg_e = self.item_embedding(pos_items), self.item_embedding(neg_items)
        pos_scores, neg_scores = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        return pos_scores, neg_scores

    def calculate_loss(self, pos_scores, neg_scores):
        bprloss = self.bprloss(pos_scores, neg_scores)
        regloss = torch.sum(torch.square(self.user_embedding.weight)) +torch.sum(torch.square(self.item_embedding.weight))
        return bprloss, regloss
