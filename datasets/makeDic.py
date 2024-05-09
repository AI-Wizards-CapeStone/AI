import os
import re
from LatexDatasets import splitLabel

def split_line(line) :
    tokens = re.findall(r'(?:\\[a-zA-Z]+|[^\s\\]+)', line)
    return tokens

if __name__ == '__main__' :
    label_path = './datasets/HME100K/train_labels.txt'
    train_dataset = []
    f = open(label_path, 'r', encoding='UTF8')
    lines = f.readlines()

    for line in lines :
        _, label = splitLabel(line)
        train_dataset.append(label)

    tokens_set = set()
    for label in train_dataset :
        tokens = split_line(label)
        tokens_set.update(tokens)

    tokens_set = sorted(tokens_set)
    latex_dict = {}
    for idx, word in enumerate(tokens_set) :
        latex_dict[word] = idx

    
    with open('./datasets/words.txt', 'w+') as f :
        for word, idx in latex_dict.items() :
            f.write(f'{word} : {idx}\n')