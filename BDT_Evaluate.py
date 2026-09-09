import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, roc_curve
import xgboost as xgb
import seaborn as sns
###################################################MODEL-LOADING#####################################################
labels=[0, 1, 2, 3, 4, 5]

var = [
    'PFP_TrackShowerScore', 'Track_EvalRatio', 'Shower_Length', 
    'Track_dEdxEnd', 'Track_Length', 'PFP_NSpacepoints', 
    'Track_DeflecAngleSD', 'Track_Concentration', 'PFP_N2DHits', 
    'Shower_InitialdEdx', 'Track_Conicalness', 'Track_CoreHaloRatio', 
    'Track_dEdxStart', 'Shower_Energy', 'Track_dEdxEndRatio', 
    'PFP_CurrentReco_Generation', 'dca', 'displament'
]
'''
np.load: load arrays from .npy, .npz, or pickled files.
'''
data = np.load('/home/lar/hadam/lar/t.npz')

# Extract features (X) and target (y);
# take |PDG| so a particle and its antiparticle collapse to one class
my_y = np.abs(data['MatchedMC_PDG'])

'''
np.unique: return the sorted unique elements of an array.
With return_counts=True it also returns how many times each value occurs
(other optional outputs, unused here, are the original and inverse indices).
'''
unique_classes, counts = np.unique(my_y, return_counts=True)

valid_classes =[11,13,22,211,321,2212 ]   # electron, muoun, photon, pi, K, protion

my_X_array = [data[i] for i in var]

'''
np.column_stack: stack 1-D arrays as columns into a 2-D array.
Each 1-D feature becomes one column; result is (n_samples, n_features).
'''
X_raw = np.column_stack(my_X_array)
mask = np.isin(my_y, valid_classes)

#NORMALISE THE CLASS DISTRIBUTION 
X = X_raw[mask]
y_filtered = my_y[mask]

for particleType, count in zip(valid_classes, range(len(valid_classes))) :
    y_filtered[y_filtered == particleType] = count

'''
train_test_split: split arrays into random train/test subsets.
stratify= preserves the per-class ratios across the splits.
'''

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

# Loading the xgboost model
best_xgb = xgb.XGBClassifier()
best_xgb.load_model('save_model.json')

y_pred = best_xgb.predict_proba(X_test) 
print(f"ypred{y_pred}") 

class_pred=(np.argmax(y_pred, axis=1))
    
print(f"class_pred:{class_pred}") 
#make a y pred and class pred

#get a pred value for each class, argmxa func

test_accuracy = accuracy_score(y_test, class_pred)
val_accuracy = accuracy_score(y_val, class_pred) 
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"\nTest Accuracy: {test_accuracy:.4f}")

print(f"Classification Report:\n{classification_report(y_val, class_pred)}")

# fresh estimator to be tuned by the randomized search

xgb_model = xgb.XGBClassifier(random_state=42, tree_method='hist') 


#################################################PLOTTING#############################################

'''
np.argsort: return the indices that would sort the array (ascending).
The [::-1] below reverses them to give descending importance order.
'''
"""
confusion_matrix: matrix C where C[i, j] is the number of samples with
true class i that were predicted as class j.
"""

X_test_df = pd.DataFrame(X_train, columns=var)

X_test_df.replace(-999, np.nan, inplace=True)
 
X_test_df.replace(-1.0, np.nan, inplace=True)

corr = X_test_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, annot_kws={"size": 6},
            cbar_kws={"shrink": .8})
plt.tight_layout()
plt.plot()
plt.savefig("correlation_matrix.png")
exit()
feature_importances = best_xgb.feature_importances_
feature_names = var 
sorted_indices = np.argsort(feature_importances)[::-1] 

#confusion matrix
cm = confusion_matrix(y_test, class_pred, normalize='pred')
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['e', 'mu', 'photon', 'pi', 'K', 'p']
).plot()

plt.savefig("my_normalised_matrix.png")

#XGB Feature Importance for Particle Reconstruction
plt.figure(figsize=(10, 5))
plt.bar(range(len(feature_importances)),
        feature_importances[sorted_indices], align="center")
plt.xticks(range(len(feature_importances)),
           np.array(feature_names)[sorted_indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importane")
plt.title("XGB Feature Importance for Particle Reconstruction") 
plt.tight_layout()
plt.savefig("my_feature_importanceplt.png")

#roc curve: Background rejection against rejection signal efficeincy 

for i in labels :
    signal_mask = (y_test == i) # [True, False, ...] [1, 0, ....]
    #y_ped = [[,,,,,], [,,,,,]]
    pid_scores = y_pred[:,i]
    #pid_scores = [0.8, 0.01, ... ]

    # ROC
    falsePositive, truePositive, _ = roc_curve(signal_mask.astype(int), pid_scores)
    bkgRejection = 1 - falsePositive
    plt.figure(figsize=(10, 5))
    plt.xlabel(" Signal Efficeincy ")
    plt.ylabel("Background Rejection")
    plt.title("ROC")    
    plt.plot(truePositive, bkgRejection, label=labels[i])
    plt.savefig(f"{labels[i]}_roc.png")

for j in labels :
    signal_mask = (y_test == j)
    pid_scores = y_pred[:,j]

    signal_scores = pid_scores[signal_mask]
    background_scores = pid_scores[~signal_mask]

    plt.figure(figsize=(10, 5))
    plt.hist(signal_scores, range=[0,1], bins=50)
    plt.hist(background_scores, range=[0,1], bins=50)
    plt.xlabel(" classification score ")
    plt.ylabel("proportion of background")
    plt.savefig(f"{labels[j]}_class_distribution.png")
    
'''
    signal_weight = np.ones_like(signal_scores.shape) / len(signal_scores)
    print(f"signal_weight{signal_weight}")
    background_weight = np.ones_like(background_scores.shape) / len(background_scores)
    print(f"background_weight{background_weight}")
    plt.hist(signal_scores, range=[0,1], bins=50, weight=signal_weight)
    plt.hist(background_scores, range=[0,1], bins=50, weight=background_weight)
'''
  







