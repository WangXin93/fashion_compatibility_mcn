import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models

from resnet import resnet50
from dgl.nn.pytorch import GatedGraphConv

import dgl

class CompatModel(nn.Module):
    def __init__(self, embed_size=1000, need_rep=False, vocabulary=None, n_steps=1, n_types=1):
        """The Pooling operation for outfit compatibility prediction.

        Args:
            embed_size: the output embedding size of the cnn model, default 1000.
            need_rep: whether to output representation of the layer before last fc
                layer, whose size is 2048. This representation can be used for
                compute the Visual Sementic Embedding (VSE) loss.
            vocabulary: the counts of words in the polyvore dataset.
        """
        super(CompatModel, self).__init__()
        cnn = resnet50(pretrained=True, need_rep=need_rep)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        self.cnn = cnn
        self.need_rep = need_rep
        self.sigmoid = nn.Sigmoid()

        # Initialize the compatibility predictor, which is a 2-layered MLP
        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)

        # Semantic embedding model
        self.sem_embedding = nn.Embedding(vocabulary, 1000)
        # Visual embedding model
        self.image_embedding = nn.Linear(2048, 1000)

        # Gated Graph Neural Network
        self.ggnn = GatedGraphConv(in_feats=embed_size, out_feats=embed_size, n_steps=n_steps, n_etypes=n_types)

        # Predictor
        self.predictor = nn.Linear(embed_size, 1)

    def forward(self, batch_g, names):
        """
        Args:
            batch_g:
            names: Description words of each item in outfit

        Return:
            out: Compatibility score
            vse_loss: Visual Semantic Loss
        """
        if self.need_rep:
            out, features, rep = self._compute_score(batch_g)
        else:
            out, features  = self._compute_score(batch_g)

        vse_loss = self._compute_vse_loss(names, rep)

        return out, vse_loss

    def _compute_vse_loss(self, names, rep):
        """ Visual semantice loss which map both visual embedding and semantic embedding 
        into a common space.

        Reference: 
        https://github.com/xthan/polyvore/blob/e0ca93b0671491564b4316982d4bfe7da17b6238/polyvore/polyvore_model_bi.py#L362
        """
        # Normalized Semantic Embedding
        padded_names = rnn_utils.pad_sequence(names, batch_first=True).to(rep.device)
        mask = torch.gt(padded_names, 0)
        cap_mask = torch.ge(mask.sum(dim=1), 2)
        semb = self.sem_embedding(padded_names)
        semb = semb * (mask.unsqueeze(dim=2)).float()
        word_lengths = mask.sum(dim=1)
        word_lengths = torch.where(
            word_lengths == 0,
            (torch.ones(semb.shape[0]).float() * 0.1).to(rep.device),
            word_lengths.float(),
        )
        semb = semb.sum(dim=1) / word_lengths.unsqueeze(dim=1)
        semb = F.normalize(semb, dim=1)

        # Normalized Visual Embedding
        vemb = F.normalize(self.image_embedding(rep), dim=1)

        # VSE Loss
        semb = torch.masked_select(semb, cap_mask.unsqueeze(dim=1))
        vemb = torch.masked_select(vemb, cap_mask.unsqueeze(dim=1))
        semb = semb.reshape([-1, 1000])
        vemb = vemb.reshape([-1, 1000])
        scores = torch.matmul(semb, vemb.transpose(0, 1))
        diagnoal = scores.diag().unsqueeze(dim=1)
        cost_s = torch.clamp(0.2 - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
        cost_im = torch.clamp(0.2 - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
        cost_s = cost_s - torch.diag(cost_s.diag())
        cost_im = cost_im - torch.diag(cost_im.diag())
        vse_loss = cost_s.sum() + cost_im.sum()
        vse_loss = vse_loss / (semb.shape[0] ** 2)

        return vse_loss

    def _compute_type_repr_loss(self, tmasks, features):
        """ Here adopt two losses to improve the type-spcified represetations.
        `tmasks_loss` expect the masks to be sparse and `features_loss` regularize
        the feature vector to be a unit vector.

        Reference:
        Conditional Similarity Networks: https://arxiv.org/abs/1603.07810
        """
        # Type embedding loss
        tmasks_loss = tmasks.norm(1) / len(tmasks)
        features_loss = features.norm(2) / np.sqrt(
            (features.shape[0] * features.shape[1])
        )
        return tmasks_loss, features_loss

    def _compute_score(self, batch_g, activate=True):
        """Extract feature vectors from input images.

        Return:
            out: the compatibility score
            features: the visual embedding of the images, we use 1000-d in all experiments
            rep: the represtions of the second last year, which is 2048-d for resnet-50 backend
        """
        images = batch_g.ndata['img']
        images = images.to(self.cnn.fc.weight.device)
        batch_size, _, _, img_size = images.shape

        if self.need_rep:
            features, *rep = self.cnn(images)
            rep = rep[-1]
        else:
            features = self.cnn(images)

        ## ggnn
        num_nodes = batch_g.number_of_nodes()
        etypes = torch.zeros(num_nodes).long()
        features = self.ggnn(batch_g, features, etypes)

        # Average pooling
        batch_g.ndata['features'] = features
        unbatch_g = dgl.unbatch(batch_g)

        # Predictor
        out = []
        for g in unbatch_g:
            feat = g.ndata['features']
            feat = feat.mean(dim=0)
            out.append(feat)
        out = torch.stack(out, dim=0)
        out = self.predictor(out)

        if activate:
            out = self.sigmoid(out)
        if self.need_rep:
            return out, features, rep
        else:
            return out, features
