import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('./datasets')

from LatexDatasets import splitLabel
from makeDic import split_line



class skip_gram_Dataset(Dataset) :
    def __init__(self, n_gram =2) :
        words_path = "./datasets/words.txt"
        labels_path = "./datasets/HME100K/train_labels.txt"
        self.words_dic = dict()
        with open(words_path, "r", encoding='UTF8') as f :
            for line in f :
                token, idx = line.strip().split(' : ')
                self.words_dic[token] = int(idx)

        self.input_data = []
        self.target_data = []
        
        with open(labels_path, "r", encoding = 'UTF8') as f:
            lines = f.readlines()
            for line in lines :
                _, label = splitLabel(line)
                label = "<sos> " + label + " <eos>"
                tokens = split_line(label)
                for i, word in enumerate(tokens) :
                    for j in range(1, n_gram+1) :
                        if (i-j) >=0 :
                            self.input_data.append(self.words_dic[word])
                            self.target_data.append(self.words_dic[tokens[i-j]])

                        if (i+j) <len(label.split()) :
                            self.input_data.append(self.words_dic[word])
                            self.target_data.append(self.words_dic[tokens[i+j]])    
        

    def __len__(self) :
        return len(self.input_data)

    def __len_word__(self) :
        return len(self.words_dic)

    def one_hot_encoding(self, integer) :
        ts = torch.zeros(self.__len_word__())
        ts[integer] = 1
        return ts

    def __getitem__(self, index) :
        return self.one_hot_encoding(self.input_data[index]), self.one_hot_encoding(self.target_data[index])
    
    


if __name__ == '__main__' :
    dataset = skip_gram_Dataset()

    """
    for i in range(10) :
        inputd, outputd = dataset.__getitem__(i), dataset.__getitem__(i)
        print(f"input data : {inputd}, output data : {outputd}")
    
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dic_len = dataset.__len_word__()
    dim = 32
    lr = 0.001
    epochs = 10
    batch_size = 20


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    W = Variable(torch.randn(dic_len, dim, device=device).float(), requires_grad=True)
    Wt = Variable(torch.randn(dim, dic_len, device=device).float(), requires_grad=True)

    optimzier = optim.Adam([W, Wt], lr = lr)
    criterion = nn.CrossEntropyLoss()

    print(W.shape, Wt.shape)
    W_path = "word2vec_W_param.pth"
    Wt_path = "word2vec_Wt_param.pth"


    for epoch in range(epochs) :
        min_loss = 30
        for input, target in dataloader :
            input = input.to(device)
            target = target.to(device)

            optimzier.zero_grad()
            output = input@W
            output = output@Wt

            loss = criterion(output, target)
            loss.backward()

            optimzier.step()

            if loss<min_loss :
                torch.save(W, W_path)
                torch.save(Wt, Wt_path)

        if epoch % 2 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')


    
    


