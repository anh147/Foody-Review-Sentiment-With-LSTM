import matplotlib.pyplot as plt
import csv
import numpy as np


fold = 0
def read_data(fold):
    i = 0
    X = []
    Y = []
    for line in open('loss_lstm/loss_fold_{}.txt'.format(fold), 'r'):
        if line[:3] == 'AUC' :
            # print(line[5:11])
            # print(line[:3])
            t = line[5:11].split(' ')[0]
            X.append(np.float32(t))
        i = i+1

    # print(np.shape(X))
    print(X)
    for i in range(np.shape(X)[0]):
        Y.append(int(i))
    return Y,X

def read_data(path):
    i = 0
    X = []
    Y = []
    for line in open(path, 'r'):
        if line[:3] != 'ACC' :
            # print(line[5:11])
            # print(line[:3])
            # t = line[5:11].split(' ')[0]
            X.append(np.float32(line))
        i = i+1

    # print(np.shape(X))
    print(X)
    for i in range(np.shape(X)[0]):
        Y.append(int(i))
    return Y,X
#fold 1
X1, Y1 = read_data('loss_lstm/train.txt')
X2, Y2 = read_data('loss_lstm/val.txt')
# X3, Y3 = read_data(2)
# X4, Y4 = read_data(3)
# X5, Y5 = read_data(4)

import matplotlib.pyplot as plt

# Creating figure and axis objects using subplots()
fig, ax = plt.subplots(figsize=[9, 7])

ax.plot(X1,Y1,marker='o', linewidth=2, label='Train')
ax.plot(X2,Y2,marker='o', linewidth=2, label='Val')
# ax.plot(X3,Y3,marker='o', linewidth=2, label='Fold 2')
# ax.plot(X4,Y4,marker='o', linewidth=2, label='Fold 3')
# ax.plot(X5,Y5,marker='o', linewidth=2, label='Fold 4')

plt.xticks(rotation=0)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.title('Loss trên tập Train và Val của mô hình LSTM')
plt.legend()
plt.show()