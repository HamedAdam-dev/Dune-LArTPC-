import numpy as np, xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# The 18 PFP/Track/Shower features fed to the classifier
var = [
    'PFP_TrackShowerScore', 'Track_EvalRatio', 'Shower_Length', 
    'Track_dEdxEnd', 'Track_Length', 'PFP_NSpacepoints', 
    'Track_DeflecAngleSD', 'Track_Concentration', 'PFP_N2DHits', 
    'Shower_InitialdEdx', 'Track_Conicalness', 'Track_CoreHaloRatio', 
    'Track_dEdxStart', 'Shower_Energy', 'Track_dEdxEndRatio', 
    'PFP_CurrentReco_Generation', 'dca', 'displament'
]

# Hyperparameter search space for the randomized search
param_grid = { 
    'n_estimators': [100, 200, 300], 
    'learning_rate': [0.01, 0.1, 0.2], 
    
    # overfitting control
    'max_depth': [3, 6, 9], 
    'min_child_weight': [1, 3, 5], 
    'subsample': [0.7, 0.85, 1.0], 
    'colsample_bytree': [0.7, 0.85, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1, 10], 
    'reg_lambda': [0.5, 0.7, 1, 1.3],
    
}

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

# Pull each named feature out of the npz into a list of 1-D arrays
my_X_array = [data[i] for i in var]

'''
np.column_stack: stack 1-D arrays as columns into a 2-D array.
Each 1-D feature becomes one column; result is (n_samples, n_features).
'''
X_raw = np.column_stack(my_X_array)
# Boolean mask selecting only events in the six target species
mask = np.isin(my_y, valid_classes)

#NORMALISE THE CLASS DISTRIBUTION 
X = X_raw[mask]
y_filtered = my_y[mask]

# Remap PDG codes to contiguous class ids 0..5 (in-place)
for particleType, count in zip(valid_classes, range(len(valid_classes))) :
    y_filtered[y_filtered == particleType] = count

'''
train_test_split: split arrays into random train/test subsets.
stratify= preserves the per-class ratios across the splits.
'''

# First split: 80% train+val, 20% held-out test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

# Second split: carve a validation set out of the 80% (0.25 * 0.8 = 20% overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

#################################################TRAINING#############################################
# Per-sample weights up-weighting kaons (class 4) 10x (computed but unused; see flag)
w_train = np.ones(y_train.shape[0])
w_train[y_train == 4] = 10.0
xgb_model = xgb.XGBClassifier(random_state=42, tree_method='hist') 
# 25 random hyperparameter draws, 3-fold CV, scored on accuracy, parallelised
grid_search = RandomizedSearchCV( 
    xgb_model, param_grid, cv=3, scoring="accuracy", 
    n_iter=25, n_jobs=-1, verbose=2, random_state=42) 

# Weights for the full train+val set actually passed to fit (kaons 10x)
w_full = np.ones(y_train_full.shape[0])
w_full[y_train_full==4]=10.0
# Fit the search on the full 80%, with kaon up-weighting
grid_search.fit(X_train_full, y_train_full, sample_weight=w_full) 
best_xgb = grid_search.best_estimator_ 
# Saving the xgboost model as JSON
best_xgb.save_model('save_model.json')

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)
