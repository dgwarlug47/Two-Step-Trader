
import torch.nn as nn
import torch


class CustomLSTM(nn.Module):
    def __init__(self, device, d, input_size, num_layers, hidden_size, dropout):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        # This hidden state and cell state are the last states that were forwarded,
        # this can be used in the validation in order to start from where you left.
        self.last_h0 = None
        self.last_c0 = None
        self.d = d
        self.fc_out = nn.Linear(self.hidden_size, self.d, bias=False)

    def forward(self, x_inp, train_or_validation):
        x_inp = x_inp.to(self.device).float()
        x_inp = x_inp[:, :, 1:]
        # train_or_validation is a variable that controls between train or validation
        # in case it is train, the input gets some zeros concatenated in the begining,
        # and after that it is forwarded to the neural net.
        # in the validation mode, the input is forwarded to the neural net without changing anything
        # Note as well that in the train mode the h0 and c0 are restarted,
        # in the validation mode the h0 and c0 are the ones use the last time.
        # x_inp should have dimensions (batch_size, seq_length, num_features)

        batch_size = x_inp.shape[0]
        num_of_features = x_inp.shape[2]
        if train_or_validation == 'train':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            x_inp = torch.cat((torch.zeros(batch_size, 1, num_of_features).to(self.device), x_inp[:, :-1, :]), dim=1).to(self.device)
        else:
            if self.last_h0 is None:
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            else:
                h0 = self.last_h0.clone()
                c0 = self.last_c0.clone()

        new_out, (final_h, final_c) = self.lstm(x_inp, (h0, c0))

        final_out = new_out
        if train_or_validation == 'validation':
            self.last_h0 = final_h
            self.last_c0 = final_c
        return self.fc_out(final_out)
        # output should have dimensions: (batch_size, seq_length, hidden_size)
