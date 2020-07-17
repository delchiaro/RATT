import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from CNIC.vocab import Vocabulary


class EncoderCNN(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # delete the last fc layer.

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        return features.reshape(features.size(0), -1)


class DecoderLSTM(nn.Module):
    def __init__(self, visual_feat_size, embed_size, hidden_size, vocab: Vocabulary, max_decode_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTM, self).__init__()
        self.full_vocab = vocab
        self.visual_embed = nn.Linear(visual_feat_size, embed_size)
        self.word_embed = nn.Embedding(len(vocab), embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))
        self.max_seq_length = max_decode_length

    def forward(self, features, target_caps, lengths, map_words=None, enforce_sorted=True):
        """Decode image feature vectors and generates captions."""
        if map_words is not None:
            target_caps = torch.tensor(map_words, dtype=torch.long).to(features.device)[target_caps]
        vemb = self.visual_embed(features)
        wemb = self.word_embed(target_caps)
        wemb = torch.cat((vemb.unsqueeze(1), wemb), 1)
        packed = pack_padded_sequence(wemb, lengths, batch_first=True, enforce_sorted=enforce_sorted)
        h_states, (h, c) = self.lstm(packed)
        logits = self.linear(h_states[0])
        return logits, h_states, (h, c)

    def sample(self, features, states=None, map_words=None):
        """Generate captions for given image features using greedy search."""
        if map_words is None:
            map_words = torch.arange(self.linear.weight.shape[0]).to(features.device)
        else:
            map_words = torch.tensor(map_words, dtype=torch.long).to(features.device)
        decoded_ids = []
        all_logits = []
        inputs = self.visual_embed(features).unsqueeze(1)
        decode_lens = torch.ones(features.shape[0], dtype=torch.long).to(features.device) * self.max_seq_length
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            logits = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            predicted_selected = logits[:, map_words].argmax(1)  # predicted: (batch_size)
            decoded_ids.append(predicted_selected)
            predicted = map_words[predicted_selected]  # translate back to full word dict
            all_logits.append(logits)
            inputs = self.word_embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
            decode_lens[((predicted == Vocabulary.END_IDX) & (decode_lens == self.max_seq_length)).nonzero()] = i + 1

        return torch.stack(decoded_ids, 1), torch.stack(all_logits, 1), decode_lens


class DecoderLSTMCell(nn.Module):
    def __init__(self, visual_feat_size, embed_size, hidden_size, vocab: Vocabulary, max_decode_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTMCell, self).__init__()
        self.full_vocab = vocab
        self.visual_embed = nn.Linear(visual_feat_size, embed_size)
        self.word_embed = nn.Embedding(len(vocab), embed_size)
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, len(vocab))
        self.max_seq_length = max_decode_length
        self.hidden_size = hidden_size
        self.embed_size = embed_size

    def forward(self, features, target_caps, lengths, map_words=None, enforce_sorted=True):
        """
        Decode a batch of image feature vectors and generates captions.
        Args:
            features: input image feature vectors (encoded with a CNN), tensor with shape (BS, feat_size).
            target_caps: target captions for the input features used for teacher forcing.
            lengths: lengths of target captions.
            map_words: map caption words respect to this list, e.g. [0, 1, 2, 3, 4, 10, 13, 15] will map words
                        10 moving to index 5, 13 to 6, 15 to 7.
            enforce_sorted: if true, target_caps must be sorted by length.

        Returns: logits, (h, c), (h_states, c_states)
        """
        if map_words is not None:
            target_caps = torch.tensor(map_words, dtype=torch.long).to(features.device)[target_caps]
        vemb = self.visual_embed(features)
        wemb = self.word_embed(target_caps)
        embeddings = torch.cat((vemb.unsqueeze(1), wemb), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=enforce_sorted)
        h = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        c = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        first = 0
        h_states = []
        c_states = []
        for bs in packed.batch_sizes:
            x = packed.data[first:first + bs]
            h, c = self.lstm_cell.forward(x[:bs], (h[:bs], c[:bs]))
            first = first + bs
            h_states.append(h)
            c_states.append(c)
        h_states = torch.cat(h_states, dim=0)
        c_states = torch.cat(c_states, dim=0)
        logits = self.linear(h_states)
        return logits, (h, c), (h_states, c_states)

    def sample(self, features, map_words=None, max_seq_len=None):
        """
        Generate captions for given image features using greedy search.
        Args:
            features: input image feature vectors (encoded with a CNN), tensor with shape (BS, feat_size).
            map_words: map caption words respect to this list, e.g. [0, 1, 2, 3, 4, 10, 13, 15] will map words
                        10 moving to index 5, 13 to 6, 15 to 7.
            max_seq_len: maximum number of word to generate per each captiond

        Returns:
        """
        if map_words is None:
            map_words = torch.arange(self.linear.weight.shape[0]).to(features.device)
        else:
            map_words = torch.tensor(map_words, dtype=torch.long).to(features.device)
        decoded_ids = []
        all_logits = []
        # inputs = features.unsqueeze(1)
        inputs = self.visual_embed(features)
        max_seq_len = self.max_seq_length if max_seq_len is None else max_seq_len
        decode_lens = torch.ones(features.shape[0], dtype=torch.long).to(features.device) * max_seq_len
        h = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        c = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        for i in range(max_seq_len):
            h, c = self.lstm_cell.forward(inputs, (h, c))  # h: (batch_size, hidden_size)
            logits = self.linear(h)  # logits:  (batch_size, vocab_size)
            predicted_selected = logits[:, map_words].argmax(1)  # predicted_selected: (batch_size)
            decoded_ids.append(predicted_selected)
            predicted = map_words[predicted_selected]  # translate back to full word dict
            all_logits.append(logits)
            inputs = self.word_embed(predicted)  # inputs: (batch_size, embed_size)
            decode_lens[((predicted == Vocabulary.END_IDX) & (decode_lens == self.max_seq_length)).nonzero()] = i + 1

        return torch.stack(decoded_ids, 1), torch.stack(all_logits, 1), decode_lens
