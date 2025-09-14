import torchvision
import matplotlib.pyplot as plt

def show_dataset(dataset,size=16,col=4):
    plt.figure(figsize=(10,10))
    for i,img in enumerate(dataset):
        if i==size:
            break
        plt.subplot(int(size/col),col,i+1)
        plt.imshow(img[0])
    plt.show()

dataset=torchvision.datasets.ImageFolder(root="ae4dc-main/anime-data/anime-data/anime-faces")
show_dataset(dataset)