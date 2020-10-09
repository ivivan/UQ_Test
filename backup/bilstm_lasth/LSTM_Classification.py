import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, recurrent_layers, dropout_p):
        super(AttentionModel, self).__init__()

        """
		Arguments
		---------
		batch_size : 
		input_dim :
		hidden_dim : 
		output_dim : 
		recurrent_layers : 
		--------
		
		"""

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.recurrent_layers = recurrent_layers
        self.dropout_p = dropout_p

        self.input_embeded = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=recurrent_layers,
                            bidirectional=True)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.output_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.label = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_sentences, batch_size=None):
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_dim)

        """

        # print(input_sentences.shape)

        # input = self.input_embeded(input_sentences)
        input = self.dropout(torch.tanh(self.input_embeded(input_sentences)))

        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
        else:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))

        # final_hidden_state.size() = (1, batch_size, hidden_size)
        output, (hidden, final_cell_state) = self.lstm(
            input, (h_0, c_0))
        # output.size() = (batch_size, num_seq, hidden_size)
        # output = output.permute(1, 0, 2)

		# tag_space=self.out2tag(out.view(len(inputs),-1))

        hidden = torch.tanh(self.output_linear(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # only pick up last layer hidden
        # final_hidden_state = torch.unbind(final_hidden_state, dim=0)[0]

        logits = self.label(hidden)

        return logits
        # return {"logits": logits, "attention": attention_scores}


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
