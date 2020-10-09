from __future__ import print_function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# class AttentionModel(torch.nn.Module):
#     def __init__(self, batch_size, input_dim, hidden_dim, output_dim, recurrent_layers, dropout_p):
#         super(AttentionModel, self).__init__()


#         self.batch_size = batch_size
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.input_dim = input_dim
#         self.recurrent_layers = recurrent_layers
#         self.dropout_p = dropout_p

#         self.input_embeded = nn.Linear(input_dim, hidden_dim//2)
#         self.dropout = nn.Dropout(dropout_p)
#         self.lstm = nn.LSTM(input_size=hidden_dim//2, hidden_size=hidden_dim, num_layers=recurrent_layers,
#                             bidirectional=True)

#         self.self_attention = nn.Sequential(
#             nn.Linear(hidden_dim*2, hidden_dim*2),
#             nn.ReLU(True),
#             nn.Linear(hidden_dim*2, 1)
#         )

#         # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
#         for names in self.lstm._all_weights:
#             for name in filter(lambda n: "bias" in n, names):
#                 bias = getattr(self.lstm, name)
#                 n = bias.size(0)
#                 start, end = n // 4, n // 2
#                 bias.data[start:end].fill_(1.)

#         self.output_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
#         self.label = nn.Linear(hidden_dim*2, output_dim)

#     def forward(self, input_sentences, batch_size=None):

#         # input = self.input_embeded(input_sentences)
#         input = self.dropout(torch.tanh(self.input_embeded(input_sentences)))

#         input = input.permute(1, 0, 2)
#         if batch_size is None:
#             h_0 = Variable(torch.zeros(2*self.recurrent_layers,
#                                        self.batch_size, self.hidden_dim).to(device))
#             c_0 = Variable(torch.zeros(2*self.recurrent_layers,
#                                        self.batch_size, self.hidden_dim).to(device))
#         else:
#             h_0 = Variable(torch.zeros(2*self.recurrent_layers,
#                                        batch_size, self.hidden_dim).to(device))
#             c_0 = Variable(torch.zeros(2*self.recurrent_layers,
#                                        batch_size, self.hidden_dim).to(device))

#         # final_hidden_state.size() = (1, batch_size, hidden_size)
#         output, (final_hidden_state, final_cell_state) = self.lstm(
#             input, (h_0, c_0))
#         # output.size() = (batch_size, num_seq, hidden_size)
#         output = output.permute(1, 0, 2)

#         attn_ene = self.self_attention(output)

#         attns = F.softmax(attn_ene.view(
#             self.batch_size, -1), dim=1).unsqueeze(2)

#         final_inputs = (output * attns).sum(dim=1)

#         logits = self.label(final_inputs)

#         return logits
#         # return {"logits": logits, "attention": attention_scores}


# self, batch_size, input_dim, hidden_dim, output_dim, recurrent_layers, dropout_p


class LSTM_attention(nn.Module):
    ''' Compose with two layers '''
    def __init__(self,lstm_hidden,bilstm_flag,data,attention_dim,attention_head,attention_dropout):
        super(LSTM_attention, self).__init__()

        self.lstm = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=1, batch_first=True, bidirectional=bilstm_flag)
        self.label_attn = multihead_attention(attention_dim, num_heads=attention_head,dropout_rate=attention_dropout)
        self.droplstm = nn.Dropout(attention_dropout)
  
        # self.lstm =self.lstm.to(device)
        # self.label_attn = self.label_attn.to(device)


    def forward(self,lstm_out,label_embs,word_seq_lengths,hidden):
        # lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        lstm_out, hidden = self.lstm(lstm_out, hidden)
        # lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        # lstm_out (seq_length * batch_size * hidden)
        label_attention_output = self.label_attn(lstm_out, label_embs, label_embs)
        # label_attention_output (batch_size, seq_len, embed_size)
        lstm_out = torch.cat([lstm_out, label_attention_output], -1)
        return lstm_out

class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0 causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        # if self.gpu:
        #     self.Q_proj = self.Q_proj.cuda()
        #     self.K_proj = self.K_proj.cuda()
        #     self.V_proj = self.V_proj.cuda()


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        return outputs