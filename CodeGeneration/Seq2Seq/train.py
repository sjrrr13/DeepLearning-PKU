from pkgutil import extend_path
import torch
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from modeling import *
from data import get_trainloader


def train(train_loader, model, criterion, optimizer, num_epochs):
    model.train()
    train_loss = []
    for epoch in range(num_epochs):
        for encoder_inputs, decoder_targets in tqdm(train_loader):
            encoder_inputs, decoder_targets = encoder_inputs.to(device), decoder_targets.to(device)
            # decoder_inputs = decoder_targets
            bos_column = torch.tensor([0] * decoder_targets.shape[0]).reshape(-1, 1).to(device)
            decoder_inputs = torch.cat((bos_column, decoder_targets[:, :-1]), dim=1)
            pred, _ = model(encoder_inputs, decoder_inputs) # (seq_len, bsz, vocab_size)
            all_loss = criterion(pred.permute(1, 2, 0), decoder_targets)    # (bsz, seq_len)
            loss = all_loss.mean()
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            train_loss.append(loss.item())
            
        print(f'[Epoch {epoch + 1}] loss: {loss:.4f}')
        torch.save(model.state_dict(), f'ckpt/{epoch+20}.pt')
        with open("log.txt", "a") as f:
            f.write(str(train_loss))
            f.write("\n")
    return train_loss


if __name__ == '__main__':
    LR = 1e-4
    EPOCHS = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    nl_vocab = torch.load("nl_vocab.pt")
    code_vocab = torch.load("code_vocab.pt")
    nl_vocab_size = len(nl_vocab)
    code_vocab_size = len(code_vocab)

    encoder = Seq2SeqEncoder(nl_vocab_size, emb_size=256, hidden_size=512, num_layers=3, dropout=0.1)
    decoder = Seq2SeqDecoder(code_vocab_size, emb_size=256, hidden_size=512, num_layers=3, dropout=0.1)
    model = Seq2SeqModel(encoder, decoder)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=2)   # Ignore padding token
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = get_trainloader(nl_vocab, code_vocab, bsz=200)
    train_loss = train(train_loader, model, criterion, optimizer, EPOCHS)
