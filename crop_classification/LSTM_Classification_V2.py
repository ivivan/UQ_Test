import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# biLSTM with soft attention


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

        self.input_embeded = nn.Linear(input_dim, hidden_dim//2)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(input_size=hidden_dim//2, hidden_size=hidden_dim, num_layers=recurrent_layers,
                            bidirectional=True)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.output_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.label = nn.Linear(hidden_dim*4, output_dim)

    def attention_net(self, lstm_output, final_state):
        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                                                                                                                                          new hidden state.

        Tensor Size :
                                                                                                                                                                                                        hidden.size() = (batch_size, hidden_size)
                                                                                                                                                                                                        attn_weights.size() = (batch_size, num_seq)
                                                                                                                                                                                                        soft_attn_weights.size() = (batch_size, num_seq)
                                                                                                                                                                                                        new_hidden_state.size() = (batch_size, hidden_size)

        """

        # hidden = final_state.squeeze(0)
        hidden = final_state
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state, soft_attn_weights
        # return new_hidden_state

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
        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (h_0, c_0))
        # output.size() = (batch_size, num_seq, hidden_size)
        output = output.permute(1, 0, 2)

        # Split in 2 tensors along dimension 0 (num_directions)
        hidden_forward, hidden_backward = torch.chunk(final_hidden_state, 2, 0)
        final_hidden_state = torch.cat(
            (hidden_forward[-1, :, :], hidden_backward[-1, :, :]), dim=1)

        attn_output, attention_scores = self.attention_net(
            output, final_hidden_state)

        final_inputs = torch.cat((attn_output, final_hidden_state), dim=1)

        logits = self.label(final_inputs)

        return logits
        # return {"logits": logits, "attention": attention_scores}


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
