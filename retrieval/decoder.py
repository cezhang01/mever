import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):

        super(Decoder, self).__init__()
        self.bce_loss = nn.functional.binary_cross_entropy

    def forward(self, emb_1, emb_2):

        score = torch.matmul(emb_1, emb_2.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device)
        loss = nn.functional.cross_entropy(score, labels)

        # score = torch.sum(torch.square(emb_1 - emb_2), dim=-1)
        # neg_indices = torch.randint(emb_1.size(0), size=[emb_1.size(0)])
        # emb_neg = emb_2[neg_indices]
        # score_neg = torch.sum(torch.square(emb_1 - emb_neg), dim=-1)
        # loss = torch.max(score - score_neg + 5, torch.zeros_like(score, dtype=score.dtype).to(score.device))
        # loss = torch.mean(loss)

        return loss