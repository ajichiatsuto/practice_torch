import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data
from torchtext.datasets import IMDB
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter
import random

batch_size = 32
ouput_size = 2
hidden_size = 256
embedding_length = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("use device: {}".format(device))

train_data, test_data = IMDB(split=("train", "test"))
train_len = len(list(train_data))
train_data, val_data = random_split(list(train_data), [int(train_len*0.8), train_len-int(train_len*0.8)])

tokenizer = get_tokenizer("basic_english")
counter = Counter()
glove = GloVe(name="6B", dim=embedding_length)

for i, (label, line) in enumerate(train_data):
    counter.update(tokenizer(line))

v = vocab(counter, specials=("<unk>","<pad>"))

def text_transform(x):
    processed_text = [token for token in tokenizer(x)]
    processed_text = glove.get_vecs_by_tokens(processed_text)
    if processed_text.shape[0] > 200:
        processed_text = processed_text[:200]
    return processed_text

def label_transform(x):
    return 1 if x == 2 else 0

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
            label_list.append(label_transform(_label))
            p_text = text_transform(_text).clone().detach()
            text_list.append(p_text)
    # if text_list.shape[0] < batch_size:
    #     pad = torch.zeros(batch_size-text_list.shape[0], 200, 300)
    #     text_list = torch.cat([text_list, pad], dim=0)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=1)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, hidden_size, output_size, vocab_size, embedding_length):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(1, self.batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, self.batch_size, self.hidden_size).to(device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(hn[-1])
        return out

criterion = nn.CrossEntropyLoss()
model = LSTMClassifier(batch_size, hidden_size, ouput_size, len(v), embedding_length).to(device)
# 必要なパラメータのみを最適化する
optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

num_epochs = 10

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    print("start epoch {}".format(epoch+1))
    model.train()
    for i, (label, line) in enumerate(train_dataloader):
        if label.shape[0] != batch_size:
            continue
        label = label.to(device)
        line = line.to(device)
        line = line.transpose(1, 0)
        optim.zero_grad()
        output = model(line)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        train_acc += (output.max(1)[1] == label).sum().item()
        print("trainning... epoch:{}, {}/{}".format(epoch+1, i+1, len(train_dataloader)), end="\r")
    avg_train_loss = train_loss / len(train_dataloader.dataset)
    avg_train_acc = train_acc / len(train_dataloader.dataset)
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    model.eval()
    with torch.no_grad():
        for i, (label, line) in enumerate(val_dataloader):
            label = label.to(device)
            line = line.to(device)
            line = line.transpose(1, 0)
            output = model(line)
            loss = criterion(output, label)
            val_loss += loss.item()
            val_acc += (output.max(1)[1] == label).sum().item()
            print("validating... epoch:{}, {}/{}".format(epoch+1, i+1, len(val_dataloader)), end="\r")
    avg_val_loss = val_loss / len(val_dataloader.dataset)
    avg_val_acc = val_acc / len(val_dataloader.dataset)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)
    print("Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}".format(epoch+1, num_epochs, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))

plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='-', label='val_loss')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss in train and val')
plt.show()

plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='-', label='val_acc')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('acc in train and val')
plt.show()

