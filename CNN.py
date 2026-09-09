import uproot
from torch.utils.data import TensorDataset
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

############################################## LOAD DATA ##############################################
# Open the ROOT file and inspect its contents
f = uproot.open("/home/lar/imawby/forHamed/ivy_0.root")
print(f.keys())
print(f.classnames())

# Read the TTree and pull every branch into numpy arrays
tree = f["ivysaur"]
tree.show()
arrays = tree.arrays(library="np")

# Two W-plane images per event (start and end of the track/shower), stacked as 2 input channels
start = arrays["StartGridW"].reshape(-1, 24, 24)
end = arrays["EndGridW"].reshape(-1, 24, 24)
images=np.stack([start,end], axis=1)
labels = np.abs(arrays["TruePDG"])  # absolute PDG so particle/antiparticle share a class

# Keep only the six particle species of interest and remap their PDG codes to 0..5
valid = [11, 13, 22, 211, 321, 2212]
mapping = {11: 0, 13: 1, 22: 2, 211: 3, 321: 4, 2212: 5}
class_names = ['e', 'mu', 'gamma', 'pi', 'K', 'p']  # e, mu, gamma, pi, K, p

# Drop events outside the six classes, then apply the remap
mask = np.isin(labels, valid)
images = images[mask]
labels = np.array([mapping[p] for p in labels[mask]], dtype=np.int64)

# Per-channel normalisation to the global max of each channel
# normalise
images = images / images.max(axis=(0, 2, 3), keepdims=True)

# 80/20 split, stratified to preserve class proportions
train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)

# Wrap arrays as tensors in a Dataset for the DataLoader
train_dataset = TensorDataset(
    torch.tensor(train_imgs, dtype=torch.float32),  # (N,1,24,24)
    torch.tensor(train_lbls, dtype=torch.long),
)
test_dataset = TensorDataset(
    torch.tensor(test_imgs, dtype=torch.float32),
    torch.tensor(test_lbls, dtype=torch.long),
)

# hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10
device = torch.device('cpu')

# Batch iterators; shuffle only the training set
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

############################################## MODEL ##############################################
# LeNet-style 2D CNN: two conv+pool blocks feeding three fully connected layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=6, kernel_size=5)   # 2 input channels (start, end)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                       # halves spatial size
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 3 * 3, 120)  # 24 -> 20 -> 10 -> 6 -> 3
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)            # 6 classes

    def forward(self, x):
        # conv -> relu -> pool, twice, then flatten into the dense head
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                        # raw logits (softmax handled by the loss)
        return x

my_CNN = CNN().to(device)
# Up-weight kaons 5x in the loss to counter their rarity in the sample
class_weights = torch.ones(6, dtype=torch.float32)
class_weights[4] = 5.0   # kaons weighted 5x, everything else 1x
class_weights = class_weights.to(device)
my_CNN_criterion = nn.CrossEntropyLoss(weight=class_weights)
my_CNN_model_optimizer = optim.Adam(my_CNN.parameters(), lr=learning_rate)

############################################## TRAIN ##############################################
# Standard supervised loop: forward, loss, backprop, step; track mean loss per epoch
for epoch in range(num_epochs):
    my_CNN.train()
    running_loss = 0.0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = my_CNN(imgs)
        loss = my_CNN_criterion(outputs, lbls)
        my_CNN_model_optimizer.zero_grad()
        loss.backward()
        my_CNN_model_optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}: loss = {running_loss / len(train_loader):.4f}")

############################################## EVAL ##############################################
# Inference on the test set; collect true and predicted labels
my_CNN.eval()
two_D_model_y_true = []
two_D_model_y_pred = []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = my_CNN(imgs)
        preds = outputs.argmax(dim=1)          # predicted class = highest logit
        two_D_model_y_true.extend(lbls.cpu().numpy())
        two_D_model_y_pred.extend(preds.cpu().numpy())

# Overall accuracy plus per-class precision/recall/F1
val_accuracy = accuracy_score(two_D_model_y_true, two_D_model_y_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Classification Report:\n{classification_report(two_D_model_y_true, two_D_model_y_pred)}")

# Confusion matrix, saved to disk
cm = confusion_matrix(two_D_model_y_true, two_D_model_y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('GridW_cnn_confusion_matrix')
