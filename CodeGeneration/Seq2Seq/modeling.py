import torch
import torch.nn as nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers=num_layers, dropout=dropout)
 
    def forward(self, encoder_inputs):
        encoder_inputs = self.embedding(encoder_inputs).permute(1, 0, 2)
        _, h_n = self.rnn(encoder_inputs)
        return h_n


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=2)
        self.rnn = nn.GRU(emb_size + hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
 
    def forward(self, decoder_inputs, encoder_states):
        decoder_inputs = self.embedding(decoder_inputs).permute(1, 0, 2)
        context = encoder_states[-1]
        context = context.repeat(decoder_inputs.shape[0], 1, 1)
        output, h_n = self.rnn(torch.cat((decoder_inputs, context), -1), encoder_states)
        logits = self.fc(output)
        return logits, h_n


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
 
    def forward(self, encoder_inputs, decoder_inputs):
        return self.decoder(decoder_inputs, self.encoder(encoder_inputs))
