import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models

from resnet import resnet50

class CompatModel(nn.Module):
    def __init__(
            self, 
            embed_size=1000, 
            need_rep=False, 
            vocabulary=None,
            vse_off=False,
            pe_off=False,
            mlp_layers=2,
            conv_feats="1234",
        ):
        """The Multi-Layered Comparison Network (MCN) for outfit compatibility
        prediction and diagnosis.

        Args:
            embed_size: the output embedding size of the cnn model, default 1000.
            need_rep: whether to output representation of the layer before last fc
                layer, whose size is 2048. This representation can be used for
                compute the Visual Sementic Embedding (VSE) loss.
            vocabulary: the counts of words in the polyvore dataset.
            vse_off: whether use visual semantic embedding.
            pe_off: whether use projected embedding.
            mlp_layers: number of mlp layers used in the last predictor part.
            conv_feats: decide which layer of conv features are used for comparision.
        """
        super(CompatModel, self).__init__()
        self.vse_off = vse_off
        self.pe_off = pe_off
        self.mlp_layers = mlp_layers
        self.conv_feats = conv_feats

        cnn = resnet50(pretrained=True, need_rep=need_rep)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        self.cnn = cnn
        self.need_rep = need_rep
        self.num_rela = 15 * len(conv_feats)
        self.bn = nn.BatchNorm1d(self.num_rela)  # 5x5 relationship matrix have 25 elements

        # Define predictor part
        if self.mlp_layers > 0:
            predictor = []
            for _ in range(self.mlp_layers-1):
                linear = nn.Linear(self.num_rela, self.num_rela)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.constant_(linear.bias, 0)
                predictor.append(linear)
                predictor.append(nn.ReLU())
            linear = nn.Linear(self.num_rela, 1)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            predictor.append(linear)
            self.predictor = nn.Sequential(*predictor)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)

        # Type specified masks
        # l1, l2, l3 is the masks for feature maps for the beginning layers
        # not suffix one is for the last layer
        self.masks = nn.Embedding(15, embed_size)
        self.masks.weight.data.normal_(0.9, 0.7)
        self.masks_l1 = nn.Embedding(15, 256)
        self.masks_l1.weight.data.normal_(0.9, 0.7)
        self.masks_l2 = nn.Embedding(15, 512)
        self.masks_l2.weight.data.normal_(0.9, 0.7)
        self.masks_l3 = nn.Embedding(15, 1024)
        self.masks_l3.weight.data.normal_(0.9, 0.7)

        # Semantic embedding model
        self.sem_embedding = nn.Embedding(vocabulary, 1000)
        # Visual embedding model
        self.image_embedding = nn.Linear(2048, 1000)

        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images, names):
        """
        Args:
            images: Outfit images with shape (N, T, C, H, W)
            names: Description words of each item in outfit

        Return:
            out: Compatibility score
            vse_loss: Visual Semantic Loss
            tmasks_loss: mask loss to encourage a sparse mask
            features_loss: regularize the feature vector to be normal
        """
        if self.need_rep:
            out, features, tmasks, rep = self._compute_score(images)
        else:
            out, features, tmasks = self._compute_score(images)

        if self.vse_off:
            vse_loss = torch.tensor(0.)
        else:
            vse_loss = self._compute_vse_loss(names, rep)
        if self.pe_off:
            tmasks_loss, features_loss = torch.tensor(0.), torch.tensor(0.)
        else:
            tmasks_loss, features_loss = self._compute_type_repr_loss(tmasks, features)

        return out, vse_loss, tmasks_loss, features_loss

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

    def _compute_score(self, images, activate=True):
        """Extract feature vectors from input images.

        Return:
            out: the compatibility score
            features: the visual embedding of the images, we use 1000-d in all experiments
            masks: the mask for type-specified embedding
            rep: the represtions of the second last year, which is 2048-d for resnet-50 backend
        """
        batch_size, item_num, _, _, img_size = images.shape
        images = torch.reshape(images, (-1, 3, img_size, img_size))
        if self.need_rep:
            features, *rep = self.cnn(images)
            rep_l1, rep_l2, rep_l3, rep_l4, rep = rep
        else:
            features = self.cnn(images)

        relations = []
        features = features.reshape(batch_size, item_num, -1)  # (32, 5, 1000)
        masks = F.relu(self.masks.weight)
        # Comparison matrix
        if "4" in self.conv_feats:
            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0,1,2,3,4], 2)):
                if self.pe_off:
                    left = F.normalize(features[:, i:i+1, :], dim=-1) # (32, 1, 1000)
                    right = F.normalize(features[:, j:j+1, :], dim=-1)
                else:
                    left = F.normalize(masks[mi] * features[:, i:i+1, :], dim=-1) # (32, 1, 1000)
                    right = F.normalize(masks[mi] * features[:, j:j+1, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze() # (32)
                relations.append(rela)

        # Comparision at Multi-Layered representations
        rep_list = []
        masks_list = []
        if "1" in self.conv_feats:
            rep_list.append(rep_l1); masks_list.append(self.masks_l1)
        if "2" in self.conv_feats:
            rep_list.append(rep_l2); masks_list.append(self.masks_l2)
        if "3" in self.conv_feats:
            rep_list.append(rep_l3); masks_list.append(self.masks_l3)
        for rep_li, masks_li in zip(rep_list, masks_list):
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, item_num, -1)
            masks_li = F.relu(masks_li.weight)
            # Enumerate all pairwise combination among the outfit then compare their features
            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0,1,2,3,4], 2)):
                if self.pe_off:
                    left = F.normalize(masks_li[mi] * rep_li[:, i:i+1, :], dim=-1) # (32, 1, 1000)
                    right = F.normalize(masks_li[mi] * rep_li[:, j:j+1, :], dim=-1)
                else:
                    left = F.normalize(masks_li[mi] * rep_li[:, i:i+1, :], dim=-1) # (32, 1, 1000)
                    right = F.normalize(masks_li[mi] * rep_li[:, j:j+1, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze() # (32)
                relations.append(rela)

        if batch_size == 1: # Inference during evaluation, which input one sample
            relations = torch.stack(relations).unsqueeze(0)
        else:
            relations = torch.stack(relations, dim=1) # (32 ,15*4)
        relations = self.bn(relations)

        # Predictor
        if self.mlp_layers == 0:
            out = relations.mean(dim=-1, keepdim=True)
        else:
            out = self.predictor(relations)

        if activate:
            out = self.sigmoid(out)
        if self.need_rep:
            return out, features, masks, rep
        else:
            return out, features, masks
