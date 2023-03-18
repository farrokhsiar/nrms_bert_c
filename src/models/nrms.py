import torch
import torch.nn as nn
from torch.nn import init
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel

from mind.main.batch import MINDBatch, ContentsEncoded


def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)

    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias)

    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def is_precomputed(x):
    return type(x) is torch.Tensor


class AdditiveAttention(nn.Module):
    def __init__(self, dim=100, r=2.):
        super().__init__()
        intermediate = int(dim * r)
        self.attn = nn.Sequential(
            nn.Linear(dim, intermediate),
            nn.Dropout(0.01),
            nn.LayerNorm(intermediate),
            nn.SiLU(),
            nn.Linear(intermediate, 1),
            nn.Softmax(1),
        )
        self.attn.apply(init_weights)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, dim]
        Returns:
            outputs, weights: [B, seq_len, dim], [B, seq_len]
        """
        w = self.attn(context).squeeze(-1)
        return torch.bmm(w.unsqueeze(1), context).squeeze(1), w


class ContentsEncoder(nn.Module):
    def __init__(
            self,
            pretrained_model_name: str,
    ):
        super().__init__()
        bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dim = bert.config.hidden_size

        self.bert = bert
        self.pooler = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.01),
            nn.LayerNorm(self.dim),
            nn.SiLU(),
        )

        self.pooler.apply(init_weights)

    def forward(self, inputs: ContentsEncoded):
        x_t = self.bert(**inputs['title'])[0]
        x_t = x_t[:, 0]
        x = self.pooler(x_t)

        return x


class NRMS(nn.Module):
    def __init__(
            self,
            pretrained_model_name: str,
            sa_pretrained_model_name: str,
    ):
        super(NRMS, self).__init__()
        self.encoder = ContentsEncoder(pretrained_model_name)

        dim = self.encoder.dim

        # Use pre trained self attention.
        # Though this weight is for token level attention, it also works as a good initialization for seq level attention.
        bert = AutoModel.from_pretrained(sa_pretrained_model_name)
        self.self_attn = bert.transformer.layer[-1]  # DistilBERT
        self.additive_attn = AdditiveAttention(dim, 0.5)

    def forward(self, inputs: MINDBatch):
        if is_precomputed(inputs['x_hist']):
            x_hist = inputs['x_hist']
        else:
            x_hist = self.encoder.forward(inputs['x_hist'])
        x_hist, mask_hist = to_dense_batch(x_hist, inputs['batch_hist'])
        x_hist = self.self_attn.forward(x_hist, attn_mask=mask_hist)[0]  # DistilBERT
        x_hist, _ = self.additive_attn(x_hist)

        if is_precomputed(inputs['x_cand']):
            x_cand = inputs['x_cand']
        else:
            x_cand = self.encoder.forward(inputs['x_cand'])
        x_cand, mask_cand = to_dense_batch(x_cand, inputs['batch_cand'])

        logits = torch.bmm(x_hist.unsqueeze(1), x_cand.permute(0, 2, 1)).squeeze(1)
        logits = logits[mask_cand]

        targets = inputs['targets']
        if targets is None:
            return logits

        if self.training:
            criterion = nn.CrossEntropyLoss()
            # criterion = LabelSmoothingCrossEntropy()
            loss = criterion(logits.reshape(targets.size(0), -1), targets)
        else:
            # In case of val, targets are multi label. It's not comparable with train.
            with torch.no_grad():
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, targets.float())

        return loss, logits

    def generate_negative_samples(self, x):
        #  creates negative samples by randomly selecting indices from the input tensor x.
        num_samples = x.size(0)

        #  generates a tensor of random integers of shape (num_samples, self.num_negative_samples)
        #  each integer is in the range [0, num_samples)
        idx = torch.randint(0, num_samples, (num_samples, self.num_negative_samples), device=x.device)

        # return tensor by indexing the input tensor x with the generated indices
        return x[idx]

    def forward_neg_sampling(self, inputs: MINDBatch):

        # inputs is a dict with 'x_hist', 'batch_hist', 'x_cand', 'batch_cand', and 'targets'.
        # 'x_hist' represents historical data
        # 'x_cand' represents candidate items
        if is_precomputed(inputs['x_hist']):
            x_hist = inputs['x_hist']
        else:
            x_hist = self.encoder.forward(inputs['x_hist'])
        x_hist, mask_hist = to_dense_batch(x_hist, inputs['batch_hist'])
        x_hist = self.self_attn.forward(x_hist, attn_mask=mask_hist)[0]  # DistilBERT
        x_hist, _ = self.additive_attn(x_hist)

        if is_precomputed(inputs['x_cand']):
            x_cand = inputs['x_cand']
        else:
            x_cand = self.encoder.forward(inputs['x_cand'])
        x_cand, mask_cand = to_dense_batch(x_cand, inputs['batch_cand'])

        # Generate negative samples
        # Create negative samples by randomly selecting indices from the input tensor x
        x_cand_neg = self.generate_negative_samples(x_cand)

        # Concatenate x_cand and x_cand_neg along the first dimension
        x_cand = torch.cat((x_cand, x_cand_neg), dim=1)

        # Compute the dot product between x_hist and x_cand
        # x_hist:(batch_size, hist_feature_size) -> after unsqueeze: (batch_size, 1, hist_feature_size)
        # x_cand: (batch_size, cand_feature_size, num_candidates) -> after permute (batch_size, num_candidates, cand_feature_size)
        # batch matrix multiplication: hist_feature_size is equal to cand_feature_size, OUTPUT: (batch_size, 1, num_candidates)
        # squeeze: reduces dimension: (batch_size, num_candidates).
        logits = torch.bmm(x_hist.unsqueeze(1), x_cand.permute(0, 2, 1)).squeeze(1)
        # Apply the mask mask_cand to the logits:
        logits = logits[mask_cand]

        targets = inputs['targets']
        if targets is None:
            return logits

        if self.training:
            criterion = nn.CrossEntropyLoss()
            # criterion = LabelSmoothingCrossEntropy()
            loss = criterion(logits.reshape(targets.size(0), -1), targets)
        else:
            # In case of val, targets are multi label. It's not comparable with train.
            with torch.no_grad():
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, targets.float())

        return loss, logits
