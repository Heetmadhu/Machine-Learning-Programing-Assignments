import numpy as np
import matplotlib.pyplot as plt
import time

##Load training data
train_data = np.load('MNIST/train_data.npy')
train_labels = np.load('MNIST/train_labels.npy')

##Load testing data
test_data = np.load('MNIST/test_data.npy')
test_labels = np.load('MNIST/test_labels.npy')

print("Training data dimension: ",np.shape(train_data))
print("Training label dimensions: ",len(train_labels))
print("Test data dimesions: ",np.shape(test_data))
print("Test labels dimensions: ",len(test_labels))

train_digit,train_count = np.unique(train_labels,return_counts=True)
print("Training set distribution:")
print(dict(zip(train_digit,train_count)))

test_digit, test_count = np.unique(test_labels,return_counts=True)
print("Test set distribution:")
print(dict(zip(test_digit,test_count)))

def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)),cmap = plt.cm.gray)
    plt.show()
    return

def vis_image(index, dataset="train"):
    if(dataset=="train"):
        show_digit(train_data[index,])
        label = train_labels[index]

    else:
        show_digit(test_data[index,])
        label = test_labels[index]

    print("Label: "+str(label))
    return

vis_image(0,"train")
vis_image(0,"test")

def squared_dist(x,y):
    return np.sum(np.square(x-y))

print("Distance from ",train_labels[4]," to ",train_labels[5]," ",squared_dist(train_data[4,],train_data[5,]))

def find_NN(x):
    distance = [squared_dist(x,train_data[i,]) for i in range(len(train_labels))]
    return np.argmin(distance)

def NN_classification(x):
    index = find_NN(x)
    return train_labels[index]

print(find_NN(test_data[100,]))
print(NN_classification(test_data[100,]))
