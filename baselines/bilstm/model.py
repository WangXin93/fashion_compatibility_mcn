import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import inception
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, need_rep=False):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.need_rep = need_rep

        cnn = inception.inception_v3(pretrained=True)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)
        self.cnn = cnn
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        if self.cnn.training:
            features, representations, _ = self.cnn(images)
        else:
            features, representations = self.cnn(images)

        if self.need_rep:
            return features, representations.squeeze()
        else:
            return features


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.1)

    def _init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(self.n_layers*self.n_directions,
                            batch_size,
                            self.hidden_size).to(device),
                torch.zeros(self.n_layers*self.n_directions,
                            batch_size,
                            self.hidden_size).to(device))

    def forward(self, input, seq_lengths):
        batch_size = input.size(0)
        hidden = self._init_hidden(batch_size)
        lstm_input = pack_padded_sequence(
            input, seq_lengths.data.cpu().numpy(), batch_first=True)

        self.lstm.flatten_parameters()
        output, hidden = self.lstm(lstm_input, hidden)

        return output[0]

class CompatModel(nn.Module):
    def __init__(self, emb_size, need_rep, vocabulary):
        super(CompatModel, self).__init__()
        self.emb_size = emb_size
        self.need_rep = need_rep
        self.encoder_cnn = EncoderCNN(emb_size, need_rep=need_rep)
        self.f_rnn = LSTMModel(emb_size, emb_size, emb_size, bidirectional=False)
        self.b_rnn = LSTMModel(emb_size, emb_size, emb_size, bidirectional=False)
        self.embedding = nn.Embedding(vocabulary, emb_size)
        self.image_embedding = nn.Linear(2048, emb_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images, names, lengths):
        """ Learn outfit compatibility by BiLSTM
        Args:
            images (torch.Tensor): (N, C, H, W) Batch of outfits are flatten into a long sequence,
                the `lsngths` args helps to distinguish them
            names (list): List of tensor. Description of each item
            lengths (list): Number of items in each outfit
        """
        # Extract visual features from image sequqnece
        emb_seqs, rep = self.encoder_cnn(images) # (20+, 512)

        # BiLSTM part
        f_loss, b_loss = self._rnn(emb_seqs, lengths)

        # VSE part
        vse_loss = self._vse_loss(names, rep, margin=0.2)

        return f_loss, b_loss, vse_loss

    def extract(self, images, lengths):
        """ Extract output of RNN part, this is for evaluation
        """
        # Extract visual features from image sequqnece
        if self.need_rep:
            emb_seqs, rep = self.encoder_cnn(images) # (20+, 512)
        else:
            emb_seqs = self.encoder_cnn(images)

        # BiLSTM part
        f_output, b_output = self._rnn(emb_seqs, lengths, batch_loss=False)

        return f_output, b_output

    def _vse_loss(self, names, rep, margin):
        """Visual Semantic Embedding loss
        Args:
            names: list of words decribing each item
            rep: deep visual embedding
            margin: margin value in margin_loss
        """
        device = next(self.parameters()).device

        # Encode normalized Semantic Embedding
        padded_names = pad_sequence(names, batch_first=True).to(device)
        mask = torch.gt(padded_names, 0)
        cap_mask = torch.ge(mask.sum(dim=1), 2)
        semb = self.embedding(padded_names)
        semb = semb * (mask.unsqueeze(dim=2)).float()
        word_lengths = mask.sum(dim=1)
        word_lengths = torch.where(
            word_lengths == 0,
            (torch.ones(semb.shape[0]).float() * 0.1).to(device),
            word_lengths.float(),
        )
        semb = semb.sum(dim=1) / word_lengths.unsqueeze(dim=1)
        semb = F.normalize(semb, dim=1)
        # Encode normalized Visual Representations
        vemb = F.normalize(self.image_embedding(rep), dim=1)

        # Compute marging loss
        semb = torch.masked_select(semb, cap_mask.unsqueeze(dim=1))
        vemb = torch.masked_select(vemb, cap_mask.unsqueeze(dim=1))
        semb = semb.reshape([-1, self.emb_size])
        vemb = vemb.reshape([-1, self.emb_size])
        scores = torch.matmul(semb, vemb.transpose(0, 1))
        diagnoal = scores.diag().unsqueeze(dim=1)
        cost_s = torch.clamp(margin - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
        cost_im = torch.clamp(margin - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
        cost_s = cost_s - torch.diag(cost_s.diag())
        cost_im = cost_im - torch.diag(cost_im.diag())
        emb_loss = cost_s.sum() + cost_im.sum()
        emb_loss = emb_loss / (semb.shape[0] ** 2)
        return emb_loss

    def _rnn(self, emb_seqs, lengths, batch_loss=True):
        """ Use LSTM to model the compatibility transition between items.

        Args:
            batch_loss(bool): If False, only output of LSTM will be returned,
                otherwise the CrossEntropy among the batch will be returned
        """
        device = next(self.parameters()).device

        # Generate input embeddings e.g. (1, 2, 3, 4)
        input_emb_list = []
        start = 0
        for length in lengths:
            input_emb_list.append(emb_seqs[start : start + length - 1])
            start += length
        f_input_embs = pad_sequence(
            input_emb_list, batch_first=True
        )  # (4, 7, 512) (1, 2, 3, 4)
        b_target_embs = pad_sequence(
            [self._flip_tensor(e) for e in input_emb_list], batch_first=True
        )  # (4, 3, 2, 1)

        # Generate target embeddings e.g. (2, 3, 4, 5)
        target_emb_list = []
        start = 0
        for length in lengths:
            target_emb_list.append(emb_seqs[start + 1 : start + length])
            start += length
        f_target_embs = pad_sequence(
            target_emb_list, batch_first=True
        )  # (2, 3, 4, 5)
        b_input_embs = pad_sequence(
            [self._flip_tensor(e) for e in target_emb_list], batch_first=True
        )  # (5, 4, 3, 2)

        seq_lengths = torch.tensor([i - 1 for i in lengths]).to(device)
        f_target_embs = pack_padded_sequence(
            f_target_embs, seq_lengths, batch_first=True
        )[0]
        b_target_embs = pack_padded_sequence(
            b_target_embs, seq_lengths, batch_first=True
        )[0]

        # Iterate through LSTM
        f_output = self.f_rnn(f_input_embs, seq_lengths)
        b_output = self.b_rnn(b_input_embs, seq_lengths)

        if not batch_loss:
            return f_output, b_output
        else:
            f_score = torch.matmul(f_output, f_target_embs.t())
            f_loss = self.criterion(f_score, torch.arange(f_score.shape[0]).to(device))
            b_score = torch.matmul(b_output, b_target_embs.t())
            b_loss = self.criterion(b_score, torch.arange(b_score.shape[0]).to(device))
            return f_loss, b_loss

    def _flip_tensor(self, tensor):
        """Flip a tensor in 0 dim for backward rnn. For example, the order of dim0 will
        be transformed from [0, 1, 2, 3] to [3, 2, 1, 0]
        """
        device = next(self.parameters()).device
        idx = [i for i in range(tensor.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(device)
        flipped_tensor = tensor.index_select(0, idx)
        return flipped_tensor
