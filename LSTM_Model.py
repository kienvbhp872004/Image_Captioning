import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.input_gate = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.output_gate = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.cell_gate  = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, input, hidden):
        h_prev,c_prev = hidden
        combined_gate = torch.cat((input,h_prev),1)
        ft = self.sigmoid(self.forget_gate(combined_gate))
        it = self.sigmoid(self.input_gate(combined_gate))
        ot = self.sigmoid(self.output_gate(combined_gate))
        c_t = self.tanh(self.cell_gate(combined_gate))
        ct = ft*c_prev + it*c_t
        ht  = ot*self.tanh(ct)
        return ht,ct

