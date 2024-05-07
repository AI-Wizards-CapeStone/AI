from torch.utils.data import Dataset
import matplotlib.pyplot as plt

"""
train 59602
val 14900
test 24607

label example

train_0.jpg	( 2 ) 2 N a O H + C u S O _ { 4 } = N a _ { 2 } S O _ { 4 } + C u ( O H ) _ { 2 } \downarrow
train_2.jpg	\angle A C B = \angle A ^ { \prime } C B ^ { \prime }
train_3.jpg	1 0 \times ( x + 5 ) = 1 3 \times ( \frac { 5 } { 7 } x + 5 )
train_4.jpg	l _ { 2 } = O B = \frac { O A } { 4 } \times 3 = \frac { 1 . 2 m } { 4 } \times 3 = 0 . 9 m

"""

label_path = "./datasets/HME100K/train_val_labels.txt"
data_path = "./datasets/HME100K/"
plt.rcParams['text.usetex'] = True

def splitLabel(label) :
    idx = gt = ""
    i = 0
    for i in range(6, len(label)) :
        if label[i] == '.' :
            break
        idx+=label[i]

    gt = label[i+4:-1]
    return int(idx), gt


class LatexDataset(Dataset) :
    def __init__(self) :    
        f = open(label_path, 'r', encoding='UTF8')
        lines = f.readlines()
        self.nums = []
        self.labels = []

        for line in lines :
            idx, label = splitLabel(line)
            self.nums.append(idx)
            self.labels.append('$$'+label+'$$')

    def __len__(self) :
        return len(self.nums)

    def __getitem__(self, index) :
        return self.nums[index], self.labels[index]
        

if __name__ == '__main__' :
    dataset = LatexDataset()
    num, label = dataset.__getitem__(5)
    latex_expression = label
    fig = plt.figure()  # Dimensions of figsize are in inches
    text = fig.text(
    x=0.5,  # x-coordinate to place the text
    y=0.5,  # y-coordinate to place the text
    s=latex_expression,
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,
    )

    plt.tight_layout
    plt.show()
    