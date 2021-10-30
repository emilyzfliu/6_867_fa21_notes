# Lecture 12: Self-supervised, Contrastive Learning
Presented by Dr. Suvrit Sra, MIT, Oct 19 2021

Notes by Emily Liu

## Motivation
Currently, neural networks can perform well on supervised learning tasks but fall short on tasks with unlabeled data. However, many machine learning problems have large amounts of unlabeled data. How do we leverage these data in our models?

## Weakly supervised learning
In some tasks, it is possible to generate metrics from the data that are correlated with the target label. Because these labels are proxies for the target as opposed to preexisting ground truths, we refer to this class of techniques as weakly supervised learning.

## Metric/similarity-driven learning
Metric learning is an example of similarity-driven learning. Intuitively, if we are able to learn the distance between two data points (how similar they are - technically unsimilar, since distance is inversely related to similarity), the points are likely to have similar metrics.
- This idea is the forerunner of "modern representation learning".

## Linear Metric Learning Setup

Let $x_1, x_2 ... x_N$ denote $N$ different training data points (vectors). We want to learn a linear representation in an **embedding** space such that **similar** points are close together (smaller distance) and **dissimilar** points are farther away.

Another way to think about this is in terms of pairs of points. If you randomly draw two data points, they can either be similar (for example, belong to the same classification category) or dissimilar.

Now, we are able to define two sets, $S$ and $D$, of similar and dissimilar pairs of points respectively:

$$
S := \{(x_i, x_j) | x_i , x_j \text{ are in the same class}\}\\
D := \{(x_i, x_j) | x_i , x_j \text{ are in different classes}\}\\
$$

We want to learn a linear transformation $x \mapsto Lx$ that preserves this similarity:
$$
x \mapsto Lx \\
||x_i - x_j|| \mapsto ||Lx_i - Lx_j||
$$

We note that in the expression

$$
||x_i - x_j||^2 = (x_i - x_j)^T L^T L (x_i - x_j) \text{,}
$$

The matrix $L^TL$ maps to a positie semidefinite matrix, which we can call $A$. This turns our task into learning $A$ such that the Mahalanobis distance
$$
d_A(x_i, x_j) = (x_i - x_j)^T A (x_i - x_j)
$$
is small for pairs $(x_i, x_j) \in S$ and large for pairs $(x_i, x_j) \in D$.

## Solving via the Geometric Approach
First, we note that a naive formulation of this task
$$
min_{A \geq 0} \sum_{(x_i, x_j) \in S} d_A(x_i, x_j) - \lambda \sum_{(x_i, x_j) \in D} d_A(x_i, x_j)
$$
fails empirically because poor scaling or choice of $D$ leads to very large $A$, giving a useless solution.

Instead, we write the minimization task as such:
$$
min_{A \geq 0} \sum_{(x_i, x_j) \in S} d_A(x_i, x_j) + \sum_{(x_i, x_j) \in D} d_{A^{-1}}(x_i, x_j)
$$

Intuitively, using the inverse of a matrix leads to the opposite behavior, so instead of subtracting $d_A$ for pairs in $D$ (and trying to maximize that number), we add $d_{A^{-1}}$ (and try to minimize the inverse, which is conceptually equivalent to maximizing the original).

We then define two matrices $S$ and $D$ (not to be confused with the sets $S$ and $D$ - will fix once I figure out how to use \mathcal on markdown) as follows:

$$
S := \sum_{(x_i, x_j) \in S} (x_i - x_j)(x_i - x_j)^T\\
D := \sum_{(x_i, x_j) \in D} (x_i - x_j)(x_i - x_j)^T\\
$$

Given this formulation, we reach an equivalent optimization problem with a closed form solution:

$$
min_{A > 0} h(A) := tr(AS) + tr(A^{-1}D)
$$

where $A$ is minimized at $A = S^-1 \#_{1/2} D$, $\#_{1/2}$ being the geometric mean $X \#_{1/2} Y = X^{1/2} (X^{-1/2}YX^{-1/2})^T X^{1/2}$.

## Self-supervised representation learning
In a task where we have lots of unlabelled data, we would likely want to use self-supervised representation learning. In self-supervised representation learning, our goal is to learn a lower-dimensional representation (which we will call $z$) of our data ($x$) such that we can train a simpler classifier on our encoded data representations. This requires less labeled data than a supervised deep network.

Since we don't have a supervised task (labels), we will need to come up with a pretext task on $m$ data points on which to train the embedding $f(x) = z$. We then finetune the model on the actual supervised task on $n$ data points ($n << m$).

For example, in vision tasks, we can use predicting relative patch locations as a pretext for object detection, visual data mining, etc. In language tasks, we can use word prediction as a pretext for question answering or sentiment analysis.

More generally, we want our pretext tasks to reflect contrastive learning. Positive pairs are close together (often generated from the same data point from the original sample) and far from a negative sample. This property should hold even with slight perturbations to the data (invariance in pretext features).

Question: why do pretrained representations help?

### Theorem
(Robinson et al 2020)

If the central condition holds and the pretext task has learning rate $O(1/m^\alpha)$, we use $m=\Omega(n^\beta)$ pretext samples, then with probability $1- \delta$, the target task has excess risk $O(\frac{\alpha \beta \log{n}+\log{1/d}}{n}+\frac{1}{n^{\alpha \beta}})$.

At a high level, this means that if the pretext task can be trained on $m$ samples, it is possible to train the target task to within a certain performance.

## A contrastive learning loss function
We want to learn similarity scores such that positive pairs score more similar to each other than negatives:

$$
min_f E_{x, x^+, \{x_i^-\}_{i=1}^N}\left[-\log \frac{e^{f(x)^T f(x^+)}}{e^{f(x)^T f(x^+)} + \sum_{i=1}^N e^{f(x)^T f(x^-_i)}}\right]
$$

In this formulation, $x$ is the anchor sample (from the original dataset), $x^+$ is a positive sample (close to $x$), and $x^-$ is a negative sample (far from $x$). We observe that similarity is large when $e^{f(x)^T f(x^+)$ is large and small when $\sum_{i=1}^N e^{f(x)^T f(x^-_i)}}$ is small. It has been shown that this loss formulation can outperform supervised pretraining (He et al 2020 - https://arxiv.org/pdf/1911.05722.pdf, Misra & van der Maaten 2020 - https://arxiv.org/pdf/1912.01991.pdf).

With this formulation, you will need to generate positive and negative examples from your anchor point. Positive examples are easily generated via a random combination of data augmentations (since a slightly perturbed version of $x$ ought to be similar to $x$). Negative examples are uniformly sampled at random from the dataset. This leads to a few problems. First, you may have false negatives if you randomly sample a data point that is actually very similar to $x$. Second, you may select "easy" negatives, points that are clearly distinct from $x$ and will not provide any new information to the model.

Removing false negative improves generalization capabilities of the model. However, since the training data is unlabelled, we are not able to identify whether or not a negative is a false negative. What we can do is use the positive and uniform samples to approximate true and false negatives.

On the other hand, easy negatives (which would have a very low similarity score) are not useful to improving model performance. What would be more useful is to sample "hard" negatives, negative samples that have closer similarity scores to positives and are more likely to be the samples the model is "wrong" on.

### Sampling negatives

Uniform: Sample from marginal $P(x^-)$.

Debiased negatives (no false negatives): Sample from $P(x^- | x, x^-$ diff class $)$.

Hard negatives: Sample from $q_\beta(x^- | x, x^-$ diff class $)$, where $q_\beta(x^-) \propto e^{\beta f(x)^T f(x^-)} \cdot P(x^-)$. $q$ is a distribution that favors harder points. $\beta$ controls the level of hardness and is a hyperparameter we can tune.

# In summary:
- Semi-supervised learning is used when only a fraction of the dataset is labelled.
- Weakly supervised learning uses a pretext task as a proxy to the target (labelled) task.
- We want similar points to be close in embedding space and dissimilar points to be farther away.
- Linear metric learning is a precursor to modern representation learning that maps points to a linear transformation, the solution to which can be found analytically.
- In self-supervised learning, it is possible to pretrain a large dataset on a pretext task and fine-tune the same model on the target task to achieve reasonable performance.
- In contrastive learning, we generate positive and negative points from an anchor point in the dataset. The positive point will be an augmented or perturbed version of the anchor point. The negative point is selected to be dissimilar from the original point. However, hard (less dissimilar) negatives improve model performance moreso than easy (very dissimilar) negatives.