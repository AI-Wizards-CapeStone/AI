import matplotlib.pyplot as plt
import os
from LatexDatasets import splitLabel

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb, amsmath}'

if __name__ == '__main__' :
    label_path = "./datasets/HME100K/train_val_labels.txt"
    gt_path = "./datasets/HME100KGT/"
    image_path = "./datasets/HME100K/"
    f = open(label_path, 'r', encoding='UTF8')
    lines = f.readlines()

    dic = {}
    for line in lines :
        idx, label = splitLabel(line)
        dic[idx] = "$$" + label + "$$"

    ##for train
    train_list = os.listdir(image_path+"train")
    flag = 0

    error_list = []
    for i, name in enumerate(train_list):
        try:
            idx = name[6:-4]
            label = dic[int(idx)]

            fig = plt.figure()
            text = fig.text(
                x=0.5,  # x-coordinate to place the text
                y=0.5,  # y-coordinate to place the text
                s=label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
            )

            plt.savefig(gt_path + 'train/train_' + idx + '.jpg')
            plt.close()
        except Exception as e:
            error_list.append(i)
    
    gt_list = os.listdir(gt_path+'train')
    with open("./datasets/HME100K/train_labels.txt", 'w+') as file :
        for gt in gt_list :
            gt_idx = int(gt[6:-4])
            file.write(gt + ' ' + dic[gt_idx][2:-2] + '\n')



    





    


        








