# docs complete.
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
try:
    from LSTM import CustomLSTM
    from berkley_transformer import Transformer
    from dual_stage_attention_for_time_series import DA
    from general_losses import wasserstein, standard_cross_entropy, jensen_shannon_entropy, real_cross_entropy
    from target_distributions import target_distributions_generator, target_distributions_getter
except:
    print('notebook_style')


class AR_Model(nn.Module):

    def __init__(self, device, data_preparation_hyper_parameters, model_hyper_parameters):
        super().__init__()
        # the input of the model has dimensions (batch, seq_len, input_channels)
        self.device = device
        self.input_channels = model_hyper_parameters['num_features']
        """ the variable hidden_size refers to the classic hidden size for the LSTM, the d_inner in the berkley transformer """
        """ and for the dual stage attention is the encodder_num_hidden and the decoder_num_hidden"""
        self.hidden_size = model_hyper_parameters['hidden_size']
        self.seq_len = data_preparation_hyper_parameters['seq_length']
        self.num_layers = model_hyper_parameters['num_layers']
        self.dropout = model_hyper_parameters['dropout']
        self.type_net = model_hyper_parameters['type_net']
        """ the lower_bound and upper bound the real returns were clipped. """
        self.lower_bound = data_preparation_hyper_parameters['lower_bound']
        self.upper_bound = data_preparation_hyper_parameters['upper_bound']
        """ the n_bins the interval [lower_bound, upper_bind] were cutted on. """
        self.d = data_preparation_hyper_parameters['n_bins']
        self.train_distribution = data_preparation_hyper_parameters['train_true_distribution']

        """ Selecting the type of neural network. """
        if self.type_net == 'LSTM':
            self.net = CustomLSTM(self.device, d=self.d, input_size=self.input_channels, num_layers=self.num_layers, hidden_size=self.hidden_size, dropout=self.dropout)
        if self.type_net == 'Transformer':
            self.net = Transformer(device=self.device, d=self.d, seq_length=self.seq_len, input_size=self.input_channels, n_layers=self.num_layers,
                                   d_model=model_hyper_parameters['d_model'], n_heads=model_hyper_parameters['n_heads'], d_inner=self.hidden_size,
                                   d_k=model_hyper_parameters['d_k'], d_v=model_hyper_parameters['d_v'],
                                   dropout=model_hyper_parameters['dropout'], mode='None').to(self.device)
        if self.type_net == 'DA':
            self.net = DA(device=self.device, T=self.seq_len, input_size=self.input_channels, encoder_num_hidden=self.hidden_size,
                          decoder_num_hidden=self.hidden_size, lower_bound=self.lower_bound, upper_bound=self.upper_bound, d=self.d).to(self.device)

        self.loss_function = model_hyper_parameters['loss_function']

        if self.type_net == 'DA':
            self.chosen_targets = 2
        else:
            self.chosen_targets = model_hyper_parameters['chosen_targets']

        """ entropy regularizer """
        self.eps = model_hyper_parameters['eps']

        """ distribution hyper_parameters """
        self.distribution_parameter = model_hyper_parameters['distribution_parameter']
        self.distribution_name = model_hyper_parameters['distribution_name']

        self.target_distributions = target_distributions_generator(self.d, self.distribution_name, self.distribution_parameter)
        """ plot the target_distribution of a specific label """
        print('training distribution of label', self.d / 2)
        fig, ax = plt.subplots()
        ax.bar(np.arange(self.d), self.target_distributions[int(self.d / 2)], color='deepskyblue')
        ax.set_title('label_distribution')
        plt.show()

        """ input_normalization part """
        self.input_normalization = model_hyper_parameters['input_normalization']
        if self.input_normalization:
            self.normalization_scalar = torch.nn.Parameter(torch.ones([1]))
            self.normalization_bias = torch.nn.Parameter(torch.zeros([1]))
        self.train_features_mean = data_preparation_hyper_parameters['train_mean']
        self.train_features_std = data_preparation_hyper_parameters['train_std']

    def loss(self, x, epoch=0):
        # x has dimensions (batch_size, seq_len, n_features + 1)
        # x[;, ;, 0] is the labels
        # x[:, :, 1:] is the features.
        # the output is a float the actual loss.
        self.epoch = epoch
        label = x[:, :, 0]
        # label has dimensions: batch x seq length
        out = self.get_neural_net_output_from_input(x, 'train')
        logits = self.get_logits_from_neural_net_output(out)
        return self.get_loss_from_neural_net_output(logits, label)

    def get_neural_net_output_from_input(self, x, train_or_validation):
        # train_or_validation is a string which is either: 'train' or 'validation'
        # x has dimensions: (bath_size, seq_len, num_features)
        # x[:, :, 0] is labels
        # x[:, :, 1:] is the features
        # returns the neural_net_output of the features.
        x = x.to(self.device)

        """ Taking care of the tuned_input_normalization """
        if self.input_normalization:
            x = x.clone()
            features = x[:, :, 1:]
            normalized_features = (features - self.train_features_mean) / self.train_features_std
            x[:, :, 1:] = normalized_features * self.normalization_scalar + self.normalization_bias

        """ Forward propagate neural network """
        return self.net(x, train_or_validation)

    def get_logits_from_neural_net_output(self, out):
        # out should have dimensions: (batch_size, seq_len, _)
        # returns the logits of the probability distribution
        logits = out
        return logits

    def get_loss_from_neural_net_output(self, logits, label):
        # logits has dimensions: [batch_size, seq_len, d]
        # label has dimensions [batch_size, seq_len]
        # returns a loss whch is just a number.
        batch_size = logits.shape[0]
        looking_at_the_last_predicted_label = False
        """ This if statement is good in order to have a glimpse of """
        """ which kind of predictions the model makes according to the input. """
        if looking_at_the_last_predicted_label and self.epoch > 1:
            print('this is your prob')
            view = logits[0][-1].softmax(dim=0)
            for index in range(view.shape[0]):
                print('estimated_probability ', index, view[index])
            for index in range(view.shape[0]):
                print('difference ', index, view[index] - self.train_distribution[index])
            print("all labels", label[0])
            print("this is the label you are looking at", label[0][-1])
            print('of the batch 0 and the last seq_length')
            print('pure_cross_entropy_loss', -torch.log(view[label[0][-1].long()]))
            input('')
            """ Selecting the targets according to the chosen_targets """
        if self.chosen_targets == 0:
            validation_label = label.reshape(-1).long()
            # validation_label has one dimensions with size: batch_size * seq_len
            validation_logits = logits.reshape(-1, self.d)
            # validation_logits has 2 dimensions with dimensions: [batch_size x seq_len,  d]
        if self.chosen_targets == 1:
            validation_label = label[:, (8 * (self.seq_len // 10)):].reshape(-1).long()
            # validation_logits has one dimensions with size: batch_size * validation_seq_len
            validation_logits = logits[:, (8 * (self.seq_len // 10)):, :].reshape(batch_size * (self.seq_len - (8 * (self.seq_len // 10))), -1)
            # validation_logits has 2 dimensions with dimensions: [batch_size x validaion_seq_len,  d]
        if self.chosen_targets == 2:
            validation_label = label[:, -1].reshape(-1).long()
            # validation_label has one dimension with size: batch_size
            validation_logits = logits[:, -1, :].reshape(batch_size, self.d)
            # validation_logits has 2 dimension with dimensions: [batch_size,  d]
        distributions = target_distributions_getter(self.target_distributions, self.device, validation_label)
        # distributions has dimensions: [batch_size, d]
        """ Choosing the appropriate loss function. And getting the loss. """
        if self.loss_function == 'jensen_shannon_entropy':
            return jensen_shannon_entropy(self.device, validation_logits, distributions, eps=self.eps).mean()
        if self.loss_function == 'standard_cross_entropy':
            return standard_cross_entropy(validation_logits, validation_label, eps=self.eps)
        if self.loss_function == 'wasserstein':
            return wasserstein(self.device, torch.softmax(validation_logits, dim=1), distributions, self.d, eps=self.eps).mean()
        if self.loss_function == 'real_cross_entropy':
            return real_cross_entropy(validation_logits, distributions, eps=self.eps)
