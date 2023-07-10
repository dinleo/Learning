import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.MNIST(root="./data/",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root="./data/",
                              train=False,
                              transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

for (X_train, y_train) in train_loader:
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.axis('off')
        plt.imshow(X_train[i][0].numpy().reshape(28, 28), cmap=plt.cm.binary)
        plt.title('Class: ' + str(y_train[i].item()))
    plt.show()
    break
