import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Tagger(nn.Module):
    def __init__(self, paras, device):
        super(Tagger, self).__init__()
        self.device = device
        self.paras = paras
        pad_index = self.paras.pad_index
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=embedding#torch.nn.Embedding to understand what's going on here
        # Entries at padding_idx do not contribute to the gradient
        self.char_embeddings = nn.Embedding(self.paras.char_vocab_size, self.paras.char_embedding_size,
                                            padding_idx=pad_index)
        self.char_size = 0
        if paras.char_type == "bilstm":

            self.char_lstm = nn.LSTM(self.paras.char_embedding_size, self.paras.char_rec_num_units,
                                     batch_first=True,
                                     bidirectional=True)

            self.char_size += 2 * self.paras.char_rec_num_units

        elif paras.char_type == "conv":

            # Characters go multiple convolutional filters
            self.char_convs = nn.ModuleList()
            for i, filter_size in enumerate(paras.char_filter_sizes):
                conv = nn.Sequential()
                padding = (self.paras.char_filter_sizes[i] - 1, 0)
                conv.add_module("word_conv_%s" % (i), nn.Conv2d(1, self.paras.char_number_of_filters[i], kernel_size=(
                    filter_size, self.paras.char_embedding_size),padding=padding))
                if self.paras.char_conv_act == "relu":
                    conv.add_module("word_conv_%s_relu" % (i), nn.ReLU())
                elif self.paras.char_conv_act == "tanh":
                    conv.add_module("word_conv_%s_tanh" % (i), nn.Tanh())
                elif self.paras.char_conv_act == "leakyrelu":
                    conv.add_module("word_conv_%s_tanh" % (i), nn.LeakyReLU())

                self.char_convs.append(conv)
                self.char_size += self.paras.char_number_of_filters[i]

        elif paras.char_type == "sum":
            self.char_size += self.paras.char_embedding_size

        self.embed_dropout = nn.Dropout(p=paras.dropout_frac)

        # Instantiate next word prediction nets
        self.lm_lstm = nn.LSTM(self.char_size, self.paras.lm_hidden_size, self.paras.lm_num_layers, batch_first=True)
        self.next_word_net = nn.Linear(self.paras.lm_hidden_size, self.paras.word_vocab_size)

        # Instantiate tag classification nets (might not get used in LM training, but still need to be instantiated)
        classification_in_size = self.char_size
        # A list of fully connected networks going from the hidden layer to each output tagset (we only have 1, POS, to worry about though)
        self.hidden2tag = nn.ModuleList()
        for i in range(len(paras.tagset_size)):
            self.hidden2tag.append(nn.Linear(classification_in_size, paras.tagset_size[i]))

    def init_lm_hidden(self):
        """Returns the initial starting hidden state and cell for LSTM LM"""
        hidden = (torch.zeros(self.paras.lm_num_layers, 1, self.paras.lm_hidden_size).to(self.device),
                  torch.zeros(self.paras.lm_num_layers, 1, self.paras.lm_hidden_size).to(self.device))
        return hidden

    def forward(self, words, lengths, hidden):
        """Each batch is a sequence of words
        """
        words_inputs_chars = Variable(torch.LongTensor(words).to(self.device))

        if self.paras.char_type == "bilstm":
            # Embeddings for words in each sentence
            emb = self.char_embeddings(words_inputs_chars)

            # Used for handling sequences of variable length
            packed_input = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)

            lstm_out, (char_hidden_out, char_cell_out) = self.char_lstm(packed_input)

            # Concatenate forward and backward hidden states into a single vector for each word
            char_hidden_out = char_hidden_out.transpose(1, 0).contiguous()
            char_hidden_out = char_hidden_out.view(char_hidden_out.size(0), -1)

            char_emb = char_hidden_out

        elif self.paras.char_type == "conv":

            # This is the 'max over time filter' where the word's representation is selected
            x = self.char_embeddings(words_inputs_chars)
            emb = x.unsqueeze(1)
            conv_outs = []
            for char_conv in self.char_convs:
                x = char_conv(emb)
                x_size = x.size()
                x = torch.max(x.view(x_size[0], x_size[1], -1), dim=2)[0]
                conv_outs.append(x)

            # Concatenates the given sequence of seq tensors in the given dimension
            char_emb = torch.cat(conv_outs, dim=1)

        elif self.paras.char_type == "sum":
            x = self.char_embeddings(words_inputs_chars)
            char_emb = torch.sum(x, dim=1)

        # Perform dropout on character embedding weights
        x = self.embed_dropout(char_emb)

        # word level classification
        tag_space = []
        if self.paras.training_type == "lm":
            # next word prediction
            x = torch.unsqueeze(x, axis=0)
            lm_out, lm_hidden = self.lm_lstm(x, hidden)
            tag_space.append(self.next_word_net(lm_out))

            return tag_space, lm_hidden

        elif self.paras.training_type == "label":
            # Word level prediction for POS tags or other morphological categories
            for classifier in self.hidden2tag:
                tag_space.append(classifier(x))

        return tag_space, None

    def get_word_embeddings(self, words, lengths):
        words_inputs_chars = Variable(torch.LongTensor(words).to(self.device))

        if self.paras.char_type == "bilstm":
            # Embeddings for words in each sentence
            emb = self.char_embeddings(words_inputs_chars)

            # Used for handling sequences of variable length
            packed_input = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)

            lstm_out, (char_hidden_out, char_cell_out) = self.char_lstm(packed_input)

            # Concatenate forward and backward hidden states into a single vector for each word
            char_hidden_out = char_hidden_out.transpose(1, 0).contiguous()
            char_hidden_out = char_hidden_out.view(char_hidden_out.size(0), -1)

            return char_hidden_out

        elif self.paras.char_type == "conv":

            # This is the 'max over time filter' where the word's representation is selected
            x = self.char_embeddings(words_inputs_chars)
            emb = x.unsqueeze(1)
            conv_outs = []
            for char_conv in self.char_convs:
                x = char_conv(emb)
                x_size = x.size()
                x = torch.max(x.view(x_size[0], x_size[1], -1), dim=2)[0]
                conv_outs.append(x)

            # Concatenates the given sequence of seq tensors in the given dimension
            return torch.cat(conv_outs, dim=1)


def init_ortho(m):
    if type(m) is nn.LSTM:
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                nn.init.constant_(bias.data, 0)
            for name in filter(lambda n: "weight" in n, names):
                weight = getattr(m, name)
                nn.init.orthogonal_(weight.data)


    elif type(m) is nn.Linear:
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias.data, 0)

    elif type(m) is nn.Embedding and m.weight.requires_grad:
        nn.init.uniform_(m.weight.data, -0.01, 0.01)
        if hasattr(m, "padding_idx"):
            torch.nn.init.constant_(m.weight.data[m.padding_idx], 0)

