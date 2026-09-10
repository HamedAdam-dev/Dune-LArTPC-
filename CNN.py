import uproot
from torch.utils.data import TensorDataset
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
#dont combine too early
########################################################################### LOAD DATA #########################################################################
# Open the ROOT file and inspect its contents
f = uproot.open("/home/lar/imawby/forHamed/ivy_0.root")
print(f.keys())
print(f.classnames())

tree = f["ivysaur"]
tree.show()
arrays = tree.arrays(library="np")

# Two W-plane images per event: one at the start, one at the end of the track/shower.
# Each is a single-channel 24x24 image they are NOT stacked into one 2-channel tensor. They are kept separate and fed through the same encoder later (see bebpop).
start = arrays["StartGridW"].reshape(-1, 1, 24, 24)   # (N, 1, 24, 24)
end = arrays["EndGridW"].reshape(-1, 1, 24, 24)        # (N, 1, 24, 24)
labels = np.abs(arrays["TruePDG"])  # absolute PDG so particle/antiparticle share a class

valid = [11, 13, 22, 211, 321, 2212]
mapping = {11: 0, 13: 1, 22: 2, 211: 3, 321: 4, 2212: 5}
class_names = ['e', 'mu', 'gamma', 'pi', 'K', 'p']  # e, mu, gamma, pi, K, p

mask = np.isin(labels, valid)
start = start[mask]
end = end[mask]
labels = np.array([mapping[p] for p in labels[mask]], dtype=np.int64)

# Per-image normalisation: divides each image by ITS OWN max pixel value (
start_max = start.max(axis=(2, 3), keepdims=True)
end_max = end.max(axis=(2, 3), keepdims=True)
start = np.divide(start, start_max, out=np.zeros_like(start), where=start_max != 0)
end = np.divide(end, end_max, out=np.zeros_like(end), where=end_max != 0)

# 80/20 split
start_train, start_test, end_train, end_test, train_lbls, test_lbls = train_test_split(
    start, end, labels, test_size=0.2, random_state=42, stratify=labels)

# Wrap arrays as tensors in a Dataset for the DataLoader
train_dataset = TensorDataset(
    torch.tensor(start_train, dtype=torch.float32),  # (N,1,24,24)
    torch.tensor(end_train, dtype=torch.float32),
    torch.tensor(train_lbls, dtype=torch.long),
)
test_dataset = TensorDataset(
    torch.tensor(start_test, dtype=torch.float32),
    torch.tensor(end_test, dtype=torch.float32),
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
###################################################################### MODEL ######################################################################

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 3 * 3, 120)  # spatial: 24 ->(conv)20 ->(pool)10 ->(conv)6 ->(pool)3
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        # x: (N,1,24,24)
        x = F.relu(self.conv1(x))   # -> (N,6,20,20)
        x = self.pool(x)            # -> (N,6,10,10)
        x = F.relu(self.conv2(x))   # -> (N,16,6,6)
        x = self.pool(x)            # -> (N,16,3,3)
        x = torch.flatten(x, 1)     # -> (N,144)
        x = F.relu(self.fc1(x))     # -> (N,120)
        x = F.relu(self.fc2(x))     # -> (N,84)  final ReLU: features are non-negative
        return x

class bebpop(nn.Module):
  
    def __init__(self):
        super().__init__()

        self.encoder = CNN() 
       
        self.head = nn.Sequential(
            nn.Linear(2 * 84, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3))
        self.out = nn.Linear(128, 6)  # final classification layer -> 6 class logits

    def forward(self, startU, endU):
        startOutput = self.encoder(startU)              # (N,84)
        endOutput = self.encoder(endU)                  # (N,84)
        combined = torch.cat([startOutput, endOutput], dim=1)  # (N,168)
        combined = self.head(combined)                  # (N,128)
        logits = self.out(combined)                     # (N,6) 
        return logits

my_CNN = bebpop().to(device)
# Up-weight kaons 5x in the loss to counter their rarity in the sample
class_weights = torch.ones(6, dtype=torch.float32)
class_weights[4] = 5.0   # index 4 == kaon; weighted 5x, everything else 1x
class_weights = class_weights.to(device)
# CrossEntropyLoss applies log-softmax internally, so the model outputs raw logits
my_CNN_criterion = nn.CrossEntropyLoss(weight=class_weights)
my_CNN_model_optimizer = optim.Adam(my_CNN.parameters(), lr=learning_rate)

####################################################################### TRAIN ##############################################
# Standard supervised loop: forward, loss, backprop, step; track mean loss per epoch
for epoch in range(num_epochs):
    my_CNN.train()                # train mode: dropout active, BatchNorm uses batch stats
    running_loss = 0.0
    for my_start_a, my_end_a, lbls in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        my_start_a, my_end_a, lbls = my_start_a.to(device), my_end_a.to(device), lbls.to(device)
        outputs = my_CNN(my_start_a, my_end_a)      # forward pass -> logits
        loss = my_CNN_criterion(outputs, lbls)
        my_CNN_model_optimizer.zero_grad()          # clear gradients from previous step
        loss.backward()                             # backprop
        my_CNN_model_optimizer.step()               # update weights
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}: loss = {running_loss / len(train_loader):.4f}")

############################################################### EVAL #####################################################################################
# Inference on the test set; collect true and predicted labels
my_CNN.eval()                     # eval mode: dropout off, BatchNorm uses running stats
two_D_model_y_true = []
two_D_model_y_pred = []

with torch.no_grad():             # no gradients needed at inference (saves memory/time)
    for my_start, my_end, lbls in test_loader:
        my_start, my_end, lbls = my_start.to(device), my_end.to(device), lbls.to(device)
        outputs = my_CNN(my_start, my_end)
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
