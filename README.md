# DUNE Particle Identification

Particle identification (PID) for the Deep Underground Neutrino Experiment (DUNE),
using gradient-boosted decision trees (XGBoost) and convolutional neural networks
(CNNs) trained on reconstructed particles. This repository contains the analysis
and performance results for both classifiers.

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [The DUNE simulation and reconstruction chain](#the-dune-simulation-and-reconstruction-chain)
- [PID methods](#pid-methods)
  - [Gradient boosted decision trees](#gradient-boosted-decision-trees)
  - [XGBoost](#xgboost)
  - [The structure score](#the-structure-score)
- [Deep learning](#deep-learning)
- [Convolutional neural networks](#convolutional-neural-networks-cnns)
- [Results: XGBoost](#results-xgboost)
- [Results: CNN](#results-cnn)
- [References](#references)

> **Note on math rendering.** Equations use `$...$` / `$$...$$`, which GitHub
> renders with MathJax. They will not render in a plain text viewer.

> **Note on figure paths.** Figures are assumed to live in a `figures/`
> directory. Two files (`End&StartGridU Report.png`,
> `End&StartGridU Matrix.png`) contain a space and an `&`; the links below
> URL-encode the spaces, but renaming these files to drop the space and `&` is
> recommended.

---

## Overview

The DUNE simulation and reconstruction chain produces reconstructed objects from
raw detector signals; PID then assigns each object a particle type. This project
takes reconstructed particles and classifies them with two independent
approaches: an XGBoost model trained on a set of hand-engineered features, and a
CNN trained on 2D start/end grid images of each particle. Performance is reported
with classification reports, confusion matrices, a ROC curve, feature importance,
and a feature correlation matrix.

## Repository structure

```
.
├── README.md
└── figures/
    ├── classification.png              # XGBoost classification report
    ├── my_normalised_matrix.png        # XGBoost confusion matrix (see flag below)
    ├── my_feature_importanceplt.png    # XGBoost feature importance
    ├── pion_roc.png                    # Pion one-vs-rest ROC curve
    ├── correlation_matrix.png          # Input-feature correlation matrix
    ├── End&StartGridU Report.png       # CNN U-plane classification report
    └── End&StartGridU Matrix.png       # CNN U-plane confusion matrix
```

---

## The DUNE simulation and reconstruction chain

The DUNE simulation and reconstruction chain has four main stages: Generation,
Geant4 propagation, detector simulation, and reconstruction [[1]](#references).

1. **Generation.** A generator takes an incoming neutrino's properties (energy,
   direction) and uses physics models and all possible interactions and their
   probabilities to pick one interaction, outputting the particles produced with
   their energies and directions.
2. **Geant4 ("G4") propagation.** Propagates those particles through the
   detector, tracking how far each travels, where it interacts, what secondaries
   it produces, and the ionisation electrons created along the way.
3. **Detector simulation.** Turns the arriving ionisation electrons into the
   electrical signal on each wire. The detector records charge on the wires as
   electrical waveforms (roughly Gaussian on the collection plane, bipolar on the
   induction planes), together with the scintillation light collected by the
   photon detectors.
4. **Reconstruction.** Recovers physics quantities from those signals. Reco1
   turns waveforms into hits, where each hit carries a wire position and a drift
   time from the trigger, so wire $\times$ time gives a 2D event view. Reco2 runs
   pattern recognition to group hits into objects. PID then assigns each object a
   particle type.

This is difficult because a drifting electron induces charge on neighbouring
wires too, on top of electronic noise. Because generation and G4 record the true
particle behind every hit, the simulation yields a truth label for each
reconstructed object, which is used to develop and tune the reconstruction
algorithms until their predictions are acceptable. The reconstructed quantities
are the physics variables inferred from the detector signals before any decision
about which particle produced them; the identified particle is what PID predicts
from those reconstructed quantities.

In short: run a Monte Carlo simulation to predict what interactions look like in
the detector, tune reconstruction on the truth labels, and then trust that the
simulation reflects reality closely enough that the same algorithms work on real
data. The identical chain runs on real data, except data enters at Reco1 because
the waveforms are measured from a real interaction, not simulated.

---

## PID methods

In machine learning the main steps are: defining the truth, in particular what
counts as signal versus background; constructing the network inputs; choosing the
network architecture; iteratively reconfiguring the network based on its
performance; and finally assessing performance on unseen data. This last point
requires splitting the data into three sets:

- **Training set** - the network learns from this; this is where the weights are fit.
- **Validation set** - never used for a gradient step; instead it gauges
  performance after each training cycle, guiding hand-tuned choices such as
  architecture and hyperparameters and exposing overfitting.
- **Test set** - because the hand-tuned choices are tuned to the validation set,
  it is no longer an unbiased measure of true performance, so the test set is
  held back and examined only once, after all training and tuning are complete.

### Gradient boosted decision trees

Decision trees classify the input into regions and then solve each region with a
simpler model. The tree is trained by taking all the features of the system,
comparing their separating power, and finding the best split; this is repeated
recursively until a stopping criterion is reached. There is only one node at the
beginning; adding a node turns a leaf into a decision node with two new leaves:

$$
\text{leaf} \longrightarrow \text{decision node} + 2\ \text{leaves}.
$$

Training data is sorted into the tree leaves, and the goal of training is to find
the most consistent leaves for the prediction. To evaluate new data, follow the
branching rules until you reach a leaf, then output a prediction based on which
training data items are in that leaf.

**Advantages:** requires no data pre-processing; offers a straightforward
interpretation of the decision; is nonlinear, so it can learn complex
correlations. **Limitations:** decision boundaries are axis-aligned boxes, so it
can struggle to capture linear or diagonal behaviour; it is a high-variance model
(small changes in the data can induce large variations in structure). Trees deep
enough to capture complex structure will also fit noise and outliers, whereas
shallow trees underfit.

**Boosting** combines several weak learners into one powerful model trained
iteratively. The trees are built sequentially, each one trained to correct the
errors left by the trees before it, and the final prediction is a weighted
combination of all of them. A BDT decision $F$ is a series of decision trees,
each depending on a set of parameters $a_m$:

$$
F(x) = \sum_m f_m(x; a_m).
$$

At each boosting step, when a new tree $f_m$ is added, its parameters $a_m$ are
optimised to minimise the loss. We generalise the classification problem as a
minimisation problem: minimise the loss function $L$, which encodes how far the
prediction is from the target. Each tree is grown optimally using the gradient
boosting method [[2]](#references). The gradient

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
$$

gives the direction and magnitude of the maximal change in $f$; following the
opposite direction reaches the minimum as quickly as possible. The step size (the
learning rate) is a compromise: too small, and convergence is very slow; too big,
and there is no convergence, jumping around the minimum.

At each step $n$ we fix the tree structure learned up to step $n-1$, compute the
gradient of the loss, train a new tree on the negative gradient $-g_m$, and add
the $n$-th tree to the sequence with a weight given by the learning rate. The
gradient of the loss at the $m$-th iteration is

$$
g_m = \left[\frac{\partial L(F, y)}{\partial F(x)}\right]_{F = F_{m-1}}.
$$

We train a tree to learn $-g_m$ and add it with weight $\eta$ (the learning
rate). The tree $t_m$ is expected to learn $-g_m$ as well as possible:

$$
F_m = F_{m-1} + t_m = F_{m-1} - \eta_m\, g_m.
$$

The central advantage is the freedom to choose the loss function $L$. Taking the
exponential loss

$$
L(F, y) = \exp\!\left(-F(x)\,y\right)
$$

reduces gradient boosting to AdaBoost [[3]](#references). AdaBoost is therefore a
special case, locked to a single loss that is sensitive to outliers. Gradient
boosting lifts this restriction. Standard choices include:

$$
\text{Mean squared error:}\qquad L(F,y) = \tfrac{1}{2}\left(y - F(x)\right)^2,
$$

$$
\text{Binomial log-likelihood loss:}\qquad L(F,y) = \ln\!\left(1 + e^{-2\,y\,F(x)}\right),
$$

and the Huber loss, which switches from a quadratic to a linear error as the
error grows, making it robust against outliers:

$$
L_\delta(y, F) =
\begin{cases}
\tfrac{1}{2}\left(y - F\right)^2 & \text{for } |y - F| \le \delta, \\
\delta\left(|y - F| - \tfrac{1}{2}\delta\right) & \text{for } |y - F| > \delta.
\end{cases}
$$

### XGBoost

The **model** in supervised learning usually refers to the mathematical
structure by which the prediction $\hat{y}_i$ is made from the input $x_i$
[[4]](#references)[[5]](#references). A common example is a linear model:

$$
\hat{y}_i = \sum_j \theta_j x_{ij}.
$$

The **parameters** are the undetermined part we learn from data (in linear
regression, the coefficients $\theta$). The objective function has two parts, a
training loss and a regularisation term:

$$
\mathrm{obj}(\theta) = L(\theta) + \Omega(\theta),
$$

where $L$ measures how predictive the model is on the training data and
$\Omega(\theta)$ controls model complexity to help avoid overfitting. XGBoost
instantiates this objective with a specific model (tree ensembles), a specific
regulariser, and a second-order optimisation step. Like a normal BDT, XGBoost is
an ensemble of trees:

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \qquad f_k \in \mathcal{F},
$$

where $K$ is the number of trees and $\mathcal{F}$ is the set of all possible
classification and regression trees (CARTs). The objective is

$$
\mathrm{obj}(\theta) = \sum_i^n l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \omega(f_k),
$$

so the total regularisation is $\Omega(\theta) = \sum_{k=1}^{K} \omega(f_k)$,
where $\omega(f_k)$ is the complexity of a single tree (defined below). Each
training step adds one new tree, so at step $t$ the ensemble contains $K = t$
trees. Learning the tree structure is much harder than a standard gradient
optimisation, and learning all trees at once is intractable, so we use an
additive strategy: fix what we have learned and add one tree at a time. Writing
the prediction at step $t$ as $\hat{y}_i^{(t)}$,

$$
\begin{aligned}
\hat{y}_i^{(0)} &= 0, \\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i), \\
\hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i) = \hat{y}_i^{(1)} + f_2(x_i), \\
&\;\;\vdots \\
\hat{y}_i^{(t)} &= \sum_{k=1}^{t} f_k(x_i) = \hat{y}_i^{(t-1)} + f_t(x_i).
\end{aligned}
$$

At each step we add the tree that minimises the objective:

$$
\begin{aligned}
\mathrm{obj}^{(t)}
&= \sum_{i=1}^{n} l\!\left(y_i, \hat{y}_i^{(t)}\right) + \sum_{k=1}^{t} \omega(f_k) \\
&= \sum_{i=1}^{n} l\!\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \omega(f_t) + \mathrm{constant}.
\end{aligned}
$$

Using mean squared error as the loss:

$$
\begin{aligned}
\mathrm{obj}^{(t)}
&= \sum_{i=1}^{n} \tfrac{1}{2}\left(y_i - \left(\hat{y}_i^{(t-1)} + f_t(x_i)\right)\right)^2 + \sum_{k=1}^{t} \omega(f_k) \\
&= \sum_{i=1}^{n} \left[ \left(\hat{y}_i^{(t-1)} - y_i\right) f_t(x_i) + \tfrac{1}{2} f_t(x_i)^2 \right] + \omega(f_t) + \mathrm{constant}.
\end{aligned}
$$

More generally, take the Taylor series of the loss to second order:

$$
\mathrm{obj}^{(t)} = \sum_{i=1}^{n} \left[ l\!\left(y_i, \hat{y}_i^{(t-1)}\right) + g_i\, f_t(x_i) + \tfrac{1}{2} h_i\, f_t^2(x_i) \right] + \omega(f_t) + \mathrm{constant},
$$

where

$$
g_i = \partial_{\hat{y}_i^{(t-1)}}\, l\!\left(y_i, \hat{y}_i^{(t-1)}\right), \qquad
h_i = \partial_{\hat{y}_i^{(t-1)}}^{2}\, l\!\left(y_i, \hat{y}_i^{(t-1)}\right).
$$

After removing constants, the objective at step $t$ is

$$
\sum_{i=1}^{n} \left[ g_i\, f_t(x_i) + \tfrac{1}{2} h_i\, f_t^2(x_i) \right] + \omega(f_t).
$$

The objective now depends only on $g_i$ and $h_i$. This is how XGBoost supports
custom loss functions: every loss (logistic regression, pairwise ranking, and so
on) is optimised with exactly the same solver that takes $g_i$ and $h_i$ as
input.

We define the complexity of the tree $\omega(f)$ by refining $f(x)$ as

$$
f_t(x) = w_{q(x)}, \qquad w \in \mathbb{R}^{T}, \qquad q : \mathbb{R}^{d} \rightarrow \{1, 2, \dots, T\},
$$

where $w$ is the vector of leaf scores, $q$ assigns each data point to a leaf, and
$T$ is the number of leaves. In XGBoost the complexity is

$$
\omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2,
$$

where $\gamma$ penalises the number of leaves and $\lambda$ penalises the squared
leaf scores. This regularisation is often overlooked in other tree packages.

### The structure score

With the reformulated tree model, the objective value with the $t$-th tree is

$$
\begin{aligned}
\mathrm{obj}^{(t)}
&\approx \sum_{i=1}^{n} \left[ g_i\, w_{q(x_i)} + \tfrac{1}{2} h_i\, w_{q(x_i)}^2 \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 \\
&= \sum_{j=1}^{T} \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \tfrac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T,
\end{aligned}
$$

where $I_j = \{\, i \mid q(x_i) = j \,\}$ is the set of indices assigned to the
$j$-th leaf. Defining $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$,

$$
\mathrm{obj}^{(t)} = \sum_{j=1}^{T} \left[ G_j w_j + \tfrac{1}{2}\left(H_j + \lambda\right) w_j^2 \right] + \gamma T.
$$

The $w_j$ are independent and each term is quadratic, so the best leaf weights and
best objective reduction for a given structure $q(x)$ are

$$
w_j^{\ast} = -\frac{G_j}{H_j + \lambda}, \qquad
\mathrm{obj}^{\ast} = -\frac{1}{2}\sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T.
$$

The last expression measures how good a structure $q(x)$ is: it acts like the
impurity measure in a decision tree, but also accounts for model complexity.
Enumerating all structures is intractable, so we grow the tree greedily, one
level at a time. For each candidate split the gain is

$$
\mathrm{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma,
$$

where $G_L, G_R$ and $H_L, H_R$ are the gradient and Hessian sums in the left and
right child nodes. The three fractions are the structure scores of the new left
leaf, the new right leaf, and the original leaf; the trailing $\gamma$ is the cost
of the extra leaf. A negative gain means the split is not worth making, which is
exactly where pruning comes from. At each node XGBoost selects the split with the
largest gain.

---

## Deep learning

Deep learning is a subfield of machine learning [[6]](#references). "Deep" refers
to successive layers of representation: the model builds up increasingly abstract
features layer by layer. A network must be trained before it can be used for
inference. A layer's weights control the transformation applied to its input, and
training finds the weights that best transform the input data into the correct
output labels. Each neuron applies

$$
\text{activation}(\mathbf{W}\cdot\text{input} + b),
$$

or, for a single neuron,

$$
y = f\left(\sum_{i=1}^{D} w_i x_i + b\right),
$$

where $x_i$ are the inputs, $D$ is the number of inputs, the weights $w_i$ set the
relative importance of each input, the bias $b$ shifts the result, and the
activation $f$ introduces the nonlinearity. Stacking these across all layers gives
the forward pass, mapping an input $X$ to output $A^{[L]}$:

$$
\begin{aligned}
A^{[0]} &= X \\
Z^{[l]} &= W^{[l]} A^{[l-1]} + b^{[l]}, && l = 1, \dots, L \\
A^{[l]} &= g^{[l]}\!\left(Z^{[l]}\right), && g^{[l]} = \mathrm{ReLU}\ (l < L),\quad g^{[L]} = \mathrm{softmax}.
\end{aligned}
$$

One **epoch** is one full pass over the training set. Training is faster if the
data is split into **batches**, updating parameters after each batch, and the
dataset is **shuffled** each epoch so the model does not learn the ordering.
Because every operation is differentiable, we can compute the gradient of the loss
with respect to every weight at once. **Backpropagation** applies the chain rule
layer by layer, propagating the output error backward to adjust weights deep in
the network. With a softmax output and categorical cross-entropy loss, the
output-layer gradient takes the simple form $dZ^{[L]} = A^{[L]} - Y$:

$$
\begin{aligned}
dZ^{[L]} &= A^{[L]} - Y \\
dW^{[l]} &= \frac{1}{m}\, dZ^{[l]} \bigl(A^{[l-1]}\bigr)^{\mathsf T}, && l = L, \dots, 1 \\
db^{[l]} &= \frac{1}{m} \sum_{\text{examples}} dZ^{[l]} \\
dZ^{[l-1]} &= \bigl(W^{[l]}\bigr)^{\mathsf T} dZ^{[l]} \odot g^{[l-1]\prime}\!\left(Z^{[l-1]}\right), && l = L, \dots, 2
\end{aligned}
$$

Finally, each weight and bias is nudged in the direction that lowers the loss,
scaled by the learning rate $\alpha$:

$$
\begin{aligned}
W^{[l]} &:= W^{[l]} - \alpha\, dW^{[l]}, && l = 1, \dots, L \\
b^{[l]} &:= b^{[l]} - \alpha\, db^{[l]}.
\end{aligned}
$$

---

## Convolutional neural networks (CNNs)

CNNs are a type of deep learning algorithm inspired by the visual cortex
hierarchy. They are built from convolution layers that use filters to extract
features from input images. Early layers learn small, local patterns; later layers
combine those into larger, more complex patterns. For a layer with $K$ filters of
size $f\times f$ acting on $C_{l-1}$ input channels,

$$
Z^{[l]}_{i,j,k} = \sum_{c=1}^{C_{l-1}} \sum_{u=1}^{f} \sum_{v=1}^{f}
W^{[l]}_{u,v,c,k}\, A^{[l-1]}_{i+u,\,j+v,\,c} + b^{[l]}_{k},
\qquad
A^{[l]}_{i,j,k} = g^{[l]}\!\left(Z^{[l]}_{i,j,k}\right),
$$

where $k$ indexes the output channel (filter) and $g^{[l]}$ is the activation. The
output feature-map size follows from filter size $f$, padding $p$, and stride $s$:

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - f}{s} \right\rfloor + 1,
$$

and identically for the width. The **pooling** layer reduces the spatial
dimensions of the feature maps, shrinking the data and making the network more
robust to small translations. Over a pooling window $\mathcal{R}_{ij}$,

$$
A^{[l]}_{i,j,k} = \max_{(u,v)\,\in\,\mathcal{R}_{ij}} A^{[l-1]}_{u,v,k}.
$$

After convolution and downsampling, the feature maps are flattened into a vector,

$$
a = \operatorname{vec}\!\left(A^{[L_{\text{conv}}]}\right) \in \mathbb{R}^{H\cdot W\cdot K},
$$

which feeds the fully connected (dense) layers ($Z^{[l]} = W^{[l]} A^{[l-1]} +
b^{[l]}$). This final network looks at all the extracted features, and training
uses the same backpropagation and gradient descent, since a convolution is itself
differentiable.

---

## Results: XGBoost

The following metrics evaluate the classifier's performance. They are built from
four counts: true positives (TP), true negatives (TN), false positives (FP), and
false negatives (FN), the four entries of the confusion matrix.

Classification accuracy is the ratio of correct predictions to all predictions:

$$
\text{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP} + \mathrm{TN} + \mathrm{FP} + \mathrm{FN}}.
$$

The next three metrics are defined per class in a one-vs-rest sense (the class of
interest as positive, all others as negative) and then averaged across classes.

**Precision** is the fraction of predicted positives that are correct, i.e. how
trustworthy a positive prediction is:

$$
\text{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}.
$$

**Recall** (the true positive rate) is the fraction of actual positives the model
recovers, i.e. how good it is at finding positives:

$$
\text{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}.
$$

**F1** is the harmonic mean of the two, high only when both are high:

$$
F_1 = 2\,\frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}.
$$

**Support** counts how many actual occurrences of each class exist in the true
data; classes with very low support yield statistically unreliable per-class
metrics. The **macro average** is the unweighted mean of a metric across the $C$
classes,

$$
\text{macro} = \frac{1}{C}\sum_{c=1}^{C} m_c,
$$

while the **weighted average** weights each class by its support $n_c$,

$$
\text{weighted} = \frac{\sum_{c=1}^{C} n_c\, m_c}{\sum_{c=1}^{C} n_c},
$$

where $m_c$ is the metric for class $c$. The macro average treats all classes
equally regardless of size, whereas the weighted average is dominated by the
well-populated classes.

![XGBoost classification report](figures/classification.png)

*Classification report for the XGBoost classifier, giving per-class precision,
recall, F1 and support.*

A confusion matrix tabulates predicted class against true class, giving deeper
insight than accuracy by showing where errors fall: not just how often the model
is wrong, but which class it confuses with which. For an $n$-class problem it is
an $n \times n$ matrix whose entry $(i,j)$ counts instances of true class $i$
predicted as class $j$, so the off-diagonal terms reveal the specific
class-to-class confusions.

<!-- FLAG: placeholder filename. The XGBoost confusion matrix is currently
     my_normalised_matrix.png; confirm and update to the real file. -->
![XGBoost confusion matrix](figures/my_normalised_matrix.png)

*Column-normalised confusion matrix for the XGBoost classifier: each column
(predicted label) is normalised to unit sum, so the diagonal gives the per-class
purity (precision). Classes are electron ($e$), muon ($\mu$), photon, pion
($\pi$), kaon ($K$) and proton ($p$). The kaon column is heavily contaminated,
reflecting its very low support.*

![XGBoost feature importance](figures/my_feature_importanceplt.png)

*Feature importance for the XGBoost classifier, ranking the input variables by
their contribution to the trained model.*

### ROC curve

![Pion ROC curve](figures/pion_roc.png)

*Signal efficiency versus background rejection for the pion classifier, taking
pions as signal and all other classes as background (one-vs-rest), traced by
scanning the cut on the classifier score. Signal efficiency is the true positive
rate $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$, the fraction of true pions tagged as
pions; background rejection is $\mathrm{TN}/(\mathrm{TN}+\mathrm{FP}) =
1-\mathrm{FPR}$, the fraction of non-pions correctly rejected. Points nearer the
top-right corner are better (high efficiency at high rejection); a classifier with
no separating power lies on the anti-diagonal, rejection $=1-$efficiency. Because
the curve is independent of the signal-to-background ratio it is well suited to
rare-event searches. The area under the curve (AUC) equals the probability that a
randomly chosen pion is assigned a higher score than a randomly chosen background
particle, and summarises the overall separation.*

### Feature correlation

Correlation matrices quantify how strongly pairs of variables are linearly
related. For our dataset we expect some features to be highly correlated, and the
matrix lets us identify and remove such redundant features. Dropping one of a
strongly correlated pair reduces training cost and yields cleaner, more stable
feature-importance rankings, with little loss in performance since the removed
feature carries almost no independent information.

![Feature correlation matrix](figures/correlation_matrix.png)

*Linear correlation matrix of the XGBoost input features.*

---

## Results: CNN

![CNN U-plane classification report](figures/End&StartGridU%20Report.png)

*Classification report for the CNN trained on the U-plane start and end grids.*

![CNN U-plane confusion matrix](figures/End&StartGridU%20Matrix.png)

*Confusion matrix for the CNN trained on the U-plane start and end grids.*

---

## References

1. DUNE Collaboration, *Deep Underground Neutrino Experiment (DUNE), Far Detector
   Technical Design Report*, Volume II: DUNE Physics, 2020.
2. J. H. Friedman, "Greedy function approximation: a gradient boosting machine",
   *Annals of Statistics* **29**(5), 1189-1232, 2001.
3. Y. Freund and R. E. Schapire, "A decision-theoretic generalization of on-line
   learning and an application to boosting", *Journal of Computer and System
   Sciences* **55**(1), 119-139, 1997.
4. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System", in
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining*, 785-794, 2016.
5. XGBoost developers, "Introduction to Boosted Trees", XGBoost documentation.
   <https://xgboost.readthedocs.io/en/stable/tutorials/model.html>
6. I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016.
