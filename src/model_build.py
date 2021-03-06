#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  model_build.py
@Contact    :  bing_zhou_barrett@hotmail.com
@License    :  (C)Copyright 2021-2022, Bing Zhou, TAMU
@Author     :  Bing Zhou (Barrett)
@Modify Time:  2021/10/6 8:22
@Version    :  1.0
@Description:  None.

Example:
    Some examples of usage here.
Attributes:
    Attribute description here.
Todo:
    * For module TODOs

'''

# import lib
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from bzrs_main.modules.ml_models.victim_finder.utils import pad_sents
import sys


def sents_to_tensor(tokenizer, sents, device):
    """

    :param tokenizer: BertTokenizer
    :param sents: list[str], list of sentences (NOTE: untokenized, continuous sentences), reversely sorted
    :param device: torch.device
    :return: sents_tensor: torch.Tensor, shape(batch_size, max_sent_length), reversely sorted
    :return: masks_tensor: torch.Tensor, shape(batch_size, max_sent_length), reversely sorted
    :return: sents_lengths: torch.Tensor, shape(batch_size), reversely sorted
    """
    tokens_list = [tokenizer.tokenize(sent) for sent in sents]
    sents_lengths = [len(tokens) for tokens in tokens_list]
    # tokens_sents_zip = zip(tokens_list, sents_lengths)
    # tokens_sents_zip = sorted(tokens_sents_zip, key=lambda x: x[1], reverse=True)
    # tokens_list, sents_lengths = zip(*tokens_sents_zip)
    tokens_list_padded = pad_sents(tokens_list, '[PAD]')
    sents_lengths = torch.tensor(sents_lengths, device=device)

    masks = []
    for tokens in tokens_list_padded:
        mask = [0 if token == '[PAD]' else 1 for token in tokens]
        masks.append(mask)
    masks_tensor = torch.tensor(masks, dtype=torch.long, device=device)
    tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    sents_tensor = torch.tensor(tokens_id_list, dtype=torch.long, device=device)

    return sents_tensor, masks_tensor, sents_lengths


class DefaultModel(nn.Module):

    def __init__(self, bert_config, device, n_class):
        """

        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param n_class: int
        """

        super(DefaultModel, self).__init__()

        self.n_class = n_class
        self.bert_config = bert_config
        self.bert = BertForSequenceClassification.from_pretrained(self.bert_config, num_labels=self.n_class)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config)
        self.device = device

    def forward(self, sents):
        """

        :param sents: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        pre_softmax = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = DefaultModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


DEFAULT_HIDDEN_LAYER_SIZE = 768  # Bert-base-uncased
DEFAULT_NUM_OF_HIDDEN_LAYERS = 13  # Bert-base-uncased


# Modify and refactor this model first and try the new approaches:
# ?????????BertConfig????????????
#########################################################################
class NonlinearModel(nn.Module):

    def __init__(self, bert_config, device, n_class, dropout_rate):
        """

        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param n_class: int
        """

        super(NonlinearModel, self).__init__()
        self.n_class = n_class
        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.linear1 = nn.Linear(DEFAULT_HIDDEN_LAYER_SIZE, self.bert_config['--hidden-size'])
        self.linear2 = nn.Linear(self.bert_config['--hidden-size'], self.bert_config['--hidden-size'])
        self.linear3 = nn.Linear(self.bert_config['--hidden-size'], self.n_class)
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.LeakyReLU()

    def forward(self, **bert_inputs):
        """
        :param bert_inputs: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """
        encoded_layers, pooled_output = self.bert(**bert_inputs)
        # encoded_layers, pooled_output = self.bert(input_ids=bert_inputs['input_ids'], attention_mask=bert_inputs['attention_mask'])
        hidden1 = self.dropout(self.activation(self.linear1(pooled_output)))
        hidden2 = self.activation(self.linear2(hidden1))
        hidden3 = self.activation(self.linear3(hidden2))

        return hidden3

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NonlinearModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, n_class=self.n_class, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class CustomBertLSTMModel(nn.Module):

    def __init__(self, bert_config, device, dropout_rate, n_class, lstm_hidden_size=None):
        """

        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param lstm_hidden_size: int
        """

        super(CustomBertLSTMModel, self).__init__()

        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if not lstm_hidden_size:
            self.lstm_hidden_size = DEFAULT_HIDDEN_LAYER_SIZE
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(DEFAULT_HIDDEN_LAYER_SIZE, self.lstm_hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, **bert_inputs):
        """

        :param sents: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        #         sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        #         encoded_layers, pooled_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor,
        #                                                   output_all_encoded_layers=False)
        encoded_layers, pooled_output = self.bert(**bert_inputs)
        mask_list = bert_inputs['attention_mask'].cpu().numpy().tolist()
        sents_lengths = [mask.count(1) for mask in mask_list]
        encoded_layers = encoded_layers.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(
            pack_padded_sequence(encoded_layers, sents_lengths, enforce_sorted=False))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertLSTMModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, lstm_hidden_size=self.lstm_hidden_size,
                         dropout_rate=self.dropout_rate, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class CustomBertLSTMAttentionModel(nn.Module):

    def __init__(self, bert_config, device, dropout_rate, n_class, lstm_hidden_size=None):
        """

        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param lstm_hidden_size: int
        """

        super(CustomBertLSTMAttentionModel, self).__init__()

        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if not lstm_hidden_size:
            self.lstm_hidden_size = DEFAULT_HIDDEN_LAYER_SIZE
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(DEFAULT_HIDDEN_LAYER_SIZE, self.lstm_hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, **bert_inputs):
        """

        :param sents: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        encoded_layers, pooled_output = self.bert(**bert_inputs)
        mask_list = bert_inputs['attention_mask'].cpu().numpy().tolist()
        sents_lengths = [mask.count(1) for mask in mask_list]
        encoded_layers = encoded_layers.permute(1, 0, 2)

        enc_hiddens, (last_hidden, last_cell) = self.lstm(
            pack_padded_sequence(encoded_layers, sents_lengths, enforce_sorted=False))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertLSTMModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, lstm_hidden_size=self.lstm_hidden_size,
                         dropout_rate=self.dropout_rate, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class CustomBertConvModel(nn.Module):

    def __init__(self, bert_config, device, dropout_rate, n_class, out_channel=16):
        """

        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param out_channel: int, NOTE: out_channel per layer of BERT
        """

        super(CustomBertConvModel, self).__init__()

        self.bert_config = bert_config
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.out_channel = out_channel
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out_channels = DEFAULT_NUM_OF_HIDDEN_LAYERS * self.out_channel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.conv = nn.Conv2d(in_channels=DEFAULT_NUM_OF_HIDDEN_LAYERS,
                              out_channels=self.out_channels,
                              kernel_size=(3, DEFAULT_HIDDEN_LAYER_SIZE),
                              groups=DEFAULT_NUM_OF_HIDDEN_LAYERS)
        self.hidden_to_softmax = nn.Linear(self.out_channels, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, **bert_inputs):
        """

        :param sents:
        :return:
        """

        encoded_layers, pooled_output, hidden_states = self.bert(**bert_inputs, output_hidden_states=True)
        encoded_stack_layer = torch.stack(hidden_states, 1)  # (batch_size, channel, max_sent_length, hidden_size)

        conv_out = self.conv(encoded_stack_layer)  # (batch_size, channel_out, some_length, 1)
        conv_out = torch.squeeze(conv_out, dim=3)  # (batch_size, channel_out, some_length)
        conv_out, _ = torch.max(conv_out, dim=2)  # (batch_size, channel_out)
        pre_softmax = self.hidden_to_softmax(conv_out)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertConvModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, out_channel=self.out_channel,
                         dropout_rate=self.dropout_rate, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


print('Settled!')
