import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x.unsqueeze(1))
        output, hidden = self.gru(x, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_lens, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)

        encoder_hidden = self.encoder(src, src_lens)
        decoder_input = tgt[:, 0]
        decoder_hidden = encoder_hidden

        for t in range(1, tgt_len):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = tgt[:, t] if teacher_force else top1

        return outputs
