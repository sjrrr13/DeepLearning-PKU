from cgi import test
from curses import nl
from re import S
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy
from torch.utils.data import TensorDataset, DataLoader
import json


def get_both_tokenizer():
    spacy_tokenizer = spacy.load("en_core_web_sm")
    def spacy_java_tokenizer(text):
        tmp = spacy_tokenizer(text)
        return [token.text for token in tmp]

    nl_tokenizer = get_tokenizer('basic_english')
    code_tokenizer = get_tokenizer(spacy_java_tokenizer)
    return nl_tokenizer, code_tokenizer


def build_vocab():
    nl_tokenizer, code_tokenizer = get_both_tokenizer()
    train_data = []
    with open("data/train.jsonl", "r") as f:
        for line in f:
            nl = json.loads(line)["nl"]
            code = json.loads(line)["code"]
            train_data.append((nl, code))

    def yield_tokens(data, idx):
        for nl, code in data:
            if idx == 0:
                yield nl_tokenizer(nl)
            else:
                yield code_tokenizer(code)

    nl_vocab = build_vocab_from_iterator(
        yield_tokens(train_data, 0), 
        min_freq=2, 
        specials=["<sos>", "<eos>", "<pad>", "<unk>"], 
        special_first=True)
    nl_vocab.set_default_index(nl_vocab["<unk>"])
    print("nl_vocab built")
    torch.save(nl_vocab, 'nl_vocab.pt')
    print("nl_vocab saved")

    code_vocab = build_vocab_from_iterator(
        yield_tokens(train_data, 1), 
        min_freq=2, 
        specials=["<sos>", "<eos>", "<pad>", "<unk>"], 
        special_first=True)
    code_vocab.set_default_index(code_vocab["<unk>"])
    print("code_vocab built")
    torch.save(code_vocab, 'code_vocab.pt')
    print("code_vocab saved")


def tokenize_file():
    """
    It takes time to tokenize all the "code" in train data, so save tokenized data to file.
    """
    nl_tokenizer, code_tokenizer = get_both_tokenizer()
    data = []
    with open("data/train.jsonl", "r") as f:
        for line in f:
            n = json.loads(line)["nl"]
            c = json.loads(line)["code"]
            data.append(((nl_tokenizer(n)), code_tokenizer(c)))

    with open("data/tokenized_train.jsonl", "w") as f:
        for n, c in data:
            f.write(json.dumps({"nl": n, "code": c}) + "\n")


def truncate_pad(data, seq_len):
    return data[:seq_len] if len(data) > seq_len else data + ['<pad>'] * (seq_len - len(data))


def tokenize(data, vocab, seq_len):
    seq_data = data + ["<eos>"]
    if seq_len != -1:
        truncated_data = truncate_pad(seq_data, seq_len)
        return [vocab[token] for token in truncated_data]
    else:
        return [vocab[token] for token in seq_data]


def get_trainloader(nl_vocab, code_vocab, bsz):
    train_nl, train_code = [], []
    with open("data/tokenized_train.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            train_nl.append(tokenize(data['nl'], nl_vocab, 592))    # max len of nl in train is 658, 592 = round(658*0.9)
            train_code.append(tokenize(data['code'], code_vocab, 145))  # max len of code in train is 162, 145 = round(162*0.9)
    train_data = TensorDataset(torch.tensor(train_nl), torch.tensor(train_code))
    return DataLoader(train_data, batch_size=bsz)


def get_testloader(nl_vocab):
    nl_tokenizer = get_tokenizer('basic_english')
    test_nl = []
    with open("data/test.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            test_nl.append(torch.tensor([tokenize(nl_tokenizer(data['nl']), nl_vocab, -1)]))
    return test_nl
