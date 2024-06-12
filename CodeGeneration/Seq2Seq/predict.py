import os
import torch
from tqdm import tqdm

from modeling import *
from data import get_testloader


def predict(test_loader, code_vocab, max_len):
    results = []
    for nl_seq in tqdm(test_loader):
        encoder_inputs = nl_seq.to(device)
        h_n = model.encoder(encoder_inputs)
        
        pred_seq = code_vocab(['<sos>'])
        for _ in range(max_len):
            decoder_inputs = torch.tensor(pred_seq[-1]).reshape(1, 1).to(device)
            pred, h_n = model.decoder(decoder_inputs, h_n)
            next_token_idx = pred.squeeze().argmax().item()
            if next_token_idx == 1: # <eos>
                break
            pred_seq.append(next_token_idx)
        pred_seq = code_vocab.lookup_tokens(pred_seq[1:])
        results.append(' '.join(pred_seq))
    return results


if __name__ == '__main__':
    filename = os.environ.get('FILENAME')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    nl_vocab = torch.load("nl_vocab.pt")
    code_vocab = torch.load("code_vocab.pt")
    nl_vocab_size = len(nl_vocab)
    code_vocab_size = len(code_vocab)

    encoder = Seq2SeqEncoder(nl_vocab_size, emb_size=256, hidden_size=512, num_layers=3, dropout=0.1)
    decoder = Seq2SeqDecoder(code_vocab_size, emb_size=256, hidden_size=512, num_layers=3, dropout=0.1)
    model = Seq2SeqModel(encoder, decoder)
    model.load_state_dict(torch.load(f'ckpt/{filename}.pt'))
    model.eval()
    model.to(device)
    
    test_loader = get_testloader(nl_vocab)
    results = predict(test_loader, code_vocab, max_len=200)
    with open(f'results/{filename}.txt', 'w') as f:
        for r in results:
            f.write(r + '\n')
