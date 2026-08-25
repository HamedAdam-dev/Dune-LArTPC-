# Dune-LArTPC-
This is repo contains all my work from my experimental particle physics group internship under the supervision of isobel mawby and thomas jones.

---

> Note: the original LaTeX preamble has been removed to make this README render nicely on GitHub. All document content, equations and figures have been preserved and reformatted to Markdown without changing the original wording.

# DUNE Particle Identification

## DUNE

The DUNE simulation and reconstruction chain has four main stages: Generation, Geant4 propagation, detector simulation, and reconstruction \cite{dune}. The Generation stage is where a generator takes [...]

## PID Methods

In machine learning the main steps are: defining the truth, in particular what counts as signal versus background; constructing the network inputs; choosing the network architecture; iteratively recon[...]

### Gradient Boosted Decision Trees

Decision trees classify the input into regions and then solve each region with a simpler model. The tree is trained by taking all the features of the system, comparing their separating power, and find[...]

$$
\text{leaf} \longrightarrow \text{decision node} + 2\ \text{leaves}.
$$

Training data is sorted into the tree leaves, and the goal of training is to find the most consistent leaves for the prediction. To evaluate new data, you follow the branching rules until you reach a [...]

Boosting combines several weak learners into one powerful model that is trained iteratively. The trees are built sequentially, each one trained to correct the errors left by the trees before it. The f[...]

The specific weighting and error-correction rule are what distinguish the variants below. A BDT decision $F$ is a series of decision trees, each depending on a set of parameters $a_m$:

$$
F(x) = \sum_m f_m(x; a_m).
$$

At each step of the boosting procedure, when we add a new tree $f_m$ to the series, we optimise its parameters $a_m$ so as to minimise the loss.

We generalise the classification problem as a minimisation problem. The target is to minimise the loss function $L$, a function that encodes how far the prediction is from the target. We can deter[...]

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
$$

gives the direction and magnitude of the maximal change in a function $f$. If we follow the opposite of this direction we reach the minimum as quickly as possible. The choice of the step during the de[...] 

At each step $n$ we fix the tree structure learned up to step $n-1$, compute the gradient of the loss, train a new tree on the negative gradient $-g_m$, and add the $n$th tree to the sequence with a w[...]

$$
g_m = \left[\frac{\partial L(F, y)}{\partial F(x)}\right]_{F = F_{m-1}}.
$$

We train a tree to learn $-g_m$ and add it to the list with a weight $\eta$ (the learning rate). The tree $t_m$ is expected to learn $-g_m$ as well as possible:

$$
F_m = F_{m-1} + t_m = F_{m-1} - \eta_m\, g_m.
$$

The central advantage of this method is the freedom to choose the loss function $L$. One particular choice recovers an older, more restrictive boosting algorithm: taking the exponential loss

$$
L(F, y) = \exp\!\left(-F(x)\,y\right)
$$

reduces gradient boosting to AdaBoost \cite{freund1997}. AdaBoost is therefore a special case, locked to a single loss that is sensitive to outliers. Gradient boosting lifts this restriction, letting [...]

$$
\text{Mean squared error:}\qquad L(F,y) = \tfrac{1}{2}\left(y - F(x)\right)^2,
$$
$$
\text{Binomial log-likelihood loss:}\qquad L(F,y) = \ln\!\left(1 + e^{-2\,y\,F(x)}\right),
$$
and the Huber loss, which switches from a quadratic to a linear error as the error grows, making it robust against outliers:

$$
L_\delta(y, F) =
\begin{cases}
\tfrac{1}{2}\left(y - F\right)^2 & \text{for } |y - F| \le \delta, \\\[4pt]
\delta\left(|y - F| - \tfrac{1}{2}\delta\right) & \text{for } |y - F| > \delta.
\end{cases}
$$

### XGBoost

The **model** in supervised learning usually refers to the mathematical structure by which the prediction $\hat{y}_i$ is made from the input $x_i$ \cite{chen2016, xgbdocs}. A common example is a [...]

$$
\hat{y}_i = \sum_j \theta_j x_{ij}.
$$

The **parameters** are the undetermined part that we need to learn from data. In linear regression problems the parameters are the coefficients $\theta$. The objective function consists of two pa[...]

$$
\mathrm{obj}(\theta) = L(\theta) + \Omega(\theta),
$$

where $L$ is the training loss (measuring how predictive the model is on the training data, e.g. the mean squared error or logistic loss introduced above) and $\Omega(\theta)$ is the total regularisa[...]

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \qquad f_k \in \mathcal{F},
$$

where $K$ is the number of trees, $f_k$ is a function in the functional space $\mathcal{F}$, and $\mathcal{F}$ is the set of all possible classification and regression trees (CARTs). The objective to [...]

$$
\mathrm{obj}(\theta) = \sum_i^n l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \omega(f_k),
$$

so the total regularisation is $\Omega(\theta) = \sum_{k=1}^{K} \omega(f_k)$, where $\omega(f_k)$ is the complexity of a single tree $f_k$ (defined below). Each training step adds one new tree, so tha[...]

$$
\begin{aligned}
\hat{y}_i^{(0)} &= 0, \\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i), \\
\hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i) = \hat{y}_i^{(1)} + f_2(x_i), \\
&\;\;\;\vdots \\
\hat{y}_i^{(t)} &= \sum_{k=1}^{t} f_k(x_i) = \hat{y}_i^{(t-1)} + f_t(x_i).
\end{aligned}
$$

At each step we add the tree that minimises our objective:

$$
\begin{aligned}
\mathrm{obj}^{(t)}
&= \sum_{i=1}^{n} l\!\left(y_i, \hat{y}_i^{(t)}\right) + \sum_{k=1}^{t} \omega(f_k) \\
&= \sum_{i=1}^{n} l\!\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \omega(f_t) + \mathrm{constant}.
\end{aligned}
$$

If we consider using the mean squared error as our loss function, the objective becomes

$$
\begin{aligned}
\mathrm{obj}^{(t)}
&= \sum_{i=1}^{n} \tfrac{1}{2}\left(y_i - \left(\hat{y}_i^{(t-1)} + f_t(x_i)\right)\right)^2 + \sum_{k=1}^{t} \omega(f_k) \\
&= \sum_{i=1}^{n} \left[ \left(\hat{y}_i^{(t-1)} - y_i\right) f_t(x_i) + \tfrac{1}{2} f_t(x_i)^2 \right] + \omega(f_t) + \mathrm{constant}.
\end{aligned}
$$

In general we can also take the Taylor series of the loss function up to second order,

$$
\mathrm{obj}^{(t)} = \sum_{i=1}^{n} \left[ l\!\left(y_i, \hat{y}_i^{(t-1)}\right) + g_i\, f_t(x_i) + \tfrac{1}{2} h_i\, f_t^2(x_i) \right] + \omega(f_t) + \mathrm{constant},
$$

where $g_i$ and $h_i$ are defined as

$$
g_i = \partial_{\hat{y}_i^{(t-1)}}\, l\!\left(y_i, \hat{y}_i^{(t-1)}\right), \qquad
h_i = \partial_{\hat{y}_i^{(t-1)}}^{2}\, l\!\left(y_i, \hat{y}_i^{(t-1)}\right).
$$

After removing all the constants, the specific objective at step $t$ becomes

$$
\sum_{i=1}^{n} \left[ g_i\, f_t(x_i) + \tfrac{1}{2} h_i\, f_t^2(x_i) \right] + \omega(f_t).
$$

The main advantage of this is that the objective now only depends on $g_i$ and $h_i$. This is how XGBoost supports custom loss functions: we can optimise every loss function, including logistic reg[...]

We must define the complexity of the tree $\omega(f)$. To do so, we refine the definition of the tree $f(x)$ as

$$
f_t(x) = w_{q(x)}, \qquad w \in \mathbb{R}^{T}, \qquad q : \mathbb{R}^{d} \rightarrow \{1, 2, \, \dots, T\}.
$$

Here $w$ is the vector of scores on the leaves, $q$ is a function assigning each data point to the corresponding leaf, and $T$ is the number of leaves. In XGBoost we define the complexity as

$$
\omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2,
$$

where $\gamma$ penalises the number of leaves and $\lambda$ penalises the squared leaf scores. More often than not, this regularisation is overlooked in other tree packages.

### The Structure Score

After reformulating the tree model, we can write the objective value with the $t$-th tree as

$$
\begin{aligned}
\mathrm{obj}^{(t)}
&\approx \sum_{i=1}^{n} \left[ g_i\, w_{q(x_i)} + \tfrac{1}{2} h_i\, w_{q(x_i)}^2 \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 \\
&= \sum_{j=1}^{T} \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \tfrac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T,
\end{aligned}
$$

where $I_j = \{\, i \mid q(x_i) = j \,\}$ is the set of indices of data points assigned to the $j$-th leaf. Defining $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$, this compresses to

$$
\mathrm{obj}^{(t)} = \sum_{j=1}^{T} \left[ G_j w_j + \tfrac{1}{2}\left(H_j + \lambda\right) w_j^2 \right] + \gamma T.
$$

The $w_j$ are independent of one another, and each term $G_j w_j + \tfrac{1}{2}(H_j + \lambda) w_j^2$ is quadratic, so the best $w_j$ for a given structure $q(x)$ and the best objective reduction we c[...]

$$
w_j^{\ast} = -\frac{G_j}{H_j + \lambda}, \qquad
\mathrm{obj}^{\ast} = -\frac{1}{2}\sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T.
$$

The last equation measures how good a tree structure $q(x)$ is: it acts like the impurity measure in a decision tree, except that it also takes the model complexity into account. Ideally we would enum[...]

$$
\mathrm{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma,
$$

where $G_L, G_R$ and $H_L, H_R$ are the gradient and Hessian sums in the left and right child nodes. The three fractions are the structure scores of the new left leaf, the new right leaf, and the orig[...]

## Deep learning

Deep learning is a subfield of machine learning \cite{goodfellow}. "Deep" refers to the use of successive layers of representation: the model builds up increasingly abstract features layer by layer.[...]

$$
\text{activation}(\mathbf{W}\cdot\text{input} + b),
$$

Equivalently, for a single neuron:

$$
y = f\left(\sum_{i=1}^{D} w_i x_i + b\right)
$$

where $x_i$ are the inputs to the neuron, $D$ is the number of inputs, and the weights $w_i$ set the relative importance of each input. The bias $b$ shifts the result, and the activation function $f$ [...]

$$
\begin{align*}
A^{[0]} &= X \\
Z^{[l]} &= W^{[l]} A^{[l-1]} + b^{[l]}
  && l = 1, \dots, L \\
A^{[l]} &= g^{[l]}\!\left(Z^{[l]}\right)
  && g^{[l]} = \mathrm{ReLU}\ (l < L),\quad g^{[L]} = \mathrm{softmax}
\end{align*}
$$

One training cycle, an epoch, is one full pass in which the network has seen the entire training dataset once. We can train faster by splitting the data into **batches** and updating the model's [...]

This is gradient descent, and the gradients themselves are obtained by propagating the output error backward through the layers. With a softmax output and categorical cross-entropy loss, the output-la[...]

$$
\begin{align*}
dZ^{[L]} &= A^{[L]} - Y \\
dW^{[l]} &= \frac{1}{m}\, dZ^{[l]} \bigl(A^{[l-1]}\bigr)^{\mathsf T}
  && l = L, \dots, 1 \\
db^{[l]} &= \frac{1}{m} \sum_{\text{examples}} dZ^{[l]} \\
dZ^{[l-1]} &= \bigl(W^{[l]}\bigr)^{\mathsf T} dZ^{[l]} \odot g^{[l-1]\prime}\!\left(Z^{[l-1]}\right)
  && l = L, \dots, 2
\end{align*}
$$

Finally, each weight and bias is nudged in the direction that lowers the loss, scaled by the learning rate $\alpha$:

$$
\begin{align*}
W^{[l]} &:= W^{[l]} - \alpha\, dW^{[l]}
  && l = 1, \dots, L \\
b^{[l]} &:= b^{[l]} - \alpha\, db^{[l]}
\end{align*}
$$

## Convolutional Neural Networks (CNNs)

CNNs are a specific type of deep learning algorithm inspired by the visual cortex hierarchy. CNNs are built from convolution layers that use filters to extract features from input images. Early convol[...]

$$
Z^{[l]}_{i,j,k} = \sum_{c=1}^{C_{l-1}} \sum_{u=1}^{f} \sum_{v=1}^{f}
W^{[l]}_{u,v,c,k}\, A^{[l-1]}_{i+u,\,j+v,\,c} + b^{[l]}_{k},
\qquad
A^{[l]}_{i,j,k} = g^{[l]}\!\left(Z^{[l]}_{i,j,k}\right),
$$

where $k$ indexes the output channel (filter) and $g^{[l]}$ is the activation. The spatial size of the output feature map follows from the filter size $f$, the padding $p$, and the stride $s$:

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - f}{s} \right\rfloor + 1,
$$

and identically for the width. The pooling layer reduces the spatial dimensions of the feature maps. This reduces the size of the data and makes the network more robust to small translations. Over a p[...]

$$
A^{[l]}_{i,j,k} = \max_{(u,v)\,\in\,\mathcal{R}_{ij}} A^{[l-1]}_{u,v,k}.
$$

Finally, the fully connected layers come after the image has been convolved with the filters and downsampled. The resulting feature maps are flattened into a vector,

$$
a = \operatorname{vec}\!\left(A^{[L_{\text{conv}}]}\right)
\in \mathbb{R}^{H\cdot W\cdot K},
$$

which is then fed into the dense layers described above ($Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$). This final network looks at all the extracted features, and training uses the same backpropagation an[...]

## Analysis


![my_feature_importanceplt.png](my_feature_importanceplt.png)
*Figure: [insert caption]*


![classification.png](classification.png)
*Figure: [insert caption]*

The following metrics evaluate the classifier's performance. They are built from four counts: true positives (TP), true negatives (TN), false positives (FP), and
false negatives (FN), the four entries of the confusion matrix.

Correlation matrices quantify how strongly pairs of variables are linearly
related. For our dataset we expect some features to be highly correlated, and the
matrix lets us identify and remove such redundant features. Dropping one of a
strongly correlated pair reduces training cost and yields cleaner, more stable
feature-importance rankings, with little loss in performance since the removed
feature carries almost no independent information.

Classification accuracy is the ratio of correct predictions to all predictions:

$$
\text{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}
{\mathrm{TP} + \mathrm{TN} + \mathrm{FP} + \mathrm{FN}}.
$$

The following three metrics are defined per class in a one-vs-rest sense (the
class of interest as positive, all others as negative) and then averaged across
classes.

Precision focuses on the positive predictions, the fraction of predicted
positives that are correct, i.e. how trustworthy a positive prediction is:

$$
\text{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}.
$$

Recall (the true positive rate) focuses on the positive class, the fraction of
actual positives the model recovers, i.e. how good it is at finding positives:

$$
\text{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}.
$$

The F1 score is the harmonic mean of the two, high only when both are high:

$$
F_1 = 2\,\frac{\text{Precision} \cdot \text{Recall}}
{\text{Precision} + \text{Recall}}.
$$

Support counts how many actual occurrences of each class exist in the true data.
Classes with very low support yield statistically unreliable per-class metrics.

The macro average is the unweighted mean of a metric across the $C$ classes,

$$
\text{macro} = \frac{1}{C}\sum_{c=1}^{C} m_c,
$$

while the weighted average weights each class by its support $n_c$,

$$
\text{weighted} = \frac{\sum_{c=1}^{C} n_c\, m_c}{\sum_{c=1}^{C} n_c},
$$

where $m_c$ is the metric for class $c$. The macro average therefore treats all
classes equally regardless of size, whereas the weighted average is dominated by
the well-populated classes.



![my_normalised_matrix.png](my_normalised_matrix.png)
*Figure: [insert caption]*

**Confusion matrix.** A table of predicted class against true class, giving
deeper insight than accuracy by showing where errors fall. Exposing not just how often the model is wrong but which class it confuses with which. For an $n$-class problem this generalises to an $n\tim[...]

### ROC Curve

![pion_roc.png](pion_roc.png)
*Figure: Pion ROC curve.*

In particle physics, a ROC curve characterises how well a classifier separates
signal from background. It plots signal efficiency (the true positive rate,
$\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$, the fraction of true signal events
correctly tagged as signal) against background acceptance (the false positive
rate, $\mathrm{FP}/(\mathrm{TN}+\mathrm{FP})$, the fraction of background events
misidentified as signal). A multivariate or neural-network/BDT classifier that
outputs a score produces a full curve as you scan the cut on that score.
Operating points nearer the top-left corner are better since they have high
signal efficiency for low background contamination. Background acceptance
$=$ signal efficiency corresponds to a classifier with no separating power, so
curves hugging it carry little information. Because the ROC is independent of the
signal-to-background ratio it is well suited to rare-event searches. The area
under the curve (AUC) is the probability that a randomly chosen signal event is
assigned a higher classifier score than a randomly chosen background event. This
is a good indicator for performance.


![correlation_matrix.png](correlation_matrix.png)
*Figure: [insert caption]*

## References

\begin{thebibliography}{9}
\bibitem{dune} DUNE Collaboration, \textit{Deep Underground Neutrino Experiment (DUNE), Far Detector Technical Design Report}, Volume~II: DUNE Physics, 2020.
\bibitem{friedman} J.~H.~Friedman, ``Greedy function approximation: a gradient boosting machine'', \textit{Annals of Statistics} \textbf{29}(5), 1189--1232, 2001.
\bibitem{freund1997} Y.~Freund and R.~E.~Schapire, ``A decision-theoretic generalization of on-line learning and an application to boosting'', \textit{Journal of Computer and System Sciences} \te[...]
\bibitem{chen2016} T.~Chen and C.~Guestrin, ``XGBoost: A Scalable Tree Boosting System'', in \textit{Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mi[...]
\bibitem{xgbdocs} XGBoost developers, ``Introduction to Boosted Trees'', XGBoost documentation. \url{https://xgboost.readthedocs.io/en/stable/tutorials/model.html}
\bibitem{goodfellow} I.~Goodfellow, Y.~Bengio and A.~Courville, \textit{Deep Learning}, MIT Press, 2016.
\end{thebibliography}
