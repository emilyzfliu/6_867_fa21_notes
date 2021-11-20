# Domain Adaptation, Covariate Shift
Presented by Dr. Tommi Jaakkola, MIT, Nov 16 2021

Notes by Emily Liu

## Motivation
Oftentimes, our training data and test data distributions are not identical. Alternatively, we train on one domain and apply our network to another domain. In either of these cases, we observe a shift in covariance (hence, covariate shift).

## Tasks and Assumptions
Machine learning tasks can be broadlly categorized into the following.

### Supervised Learning
In a supervised learning task, our sample space and our task are from the same distribution, and we know both $P(x)$ and $P(y|x)$.

### Semisupervised Learning
In semisupervised learning, we have a small number $n$ labeled source samples and a large number $m$ of unlabeled samples. These samples all come from the same distribution.

### Multitask Learning
Multitask learning consists of multiple samples $(x_i, y_i)$ all drawn from different but related distributions.

Multitask learning assumes that we can rely on the same set of basis features for prediction, even if the tasks themselves are different. If the tasks are related, pooling the data for learning the most complex parts is computationally efficient. (However, note that downstream tasks may share features but use them in different ways. For example, the word "small" may be used negatively in the context of reviewing a hotel room but positively in the context of reviewing a cellphone).

### Supervised Domain Adaptation
Our task is in a different domain than the source data, but we have labelled samples in both the source domain and the target domain. (There are many more labelled samples in the source domain than in the target domain).

### Unsupervised Domain Adaptation
Our task is in a different domain than the source data, but we have labelled samples in the source domain only. There are no labels in the target domain, meaning our task becomes drawing $(x_i, y_i) \sim P_S(y|x) P_T(x)$.

It is in unsupervised domain adaptation that we observe covariate shift.

## Covariate Shift
We define our source and target data as follows:
$$
S_n = \{(x_i, y_i), i=1...n\}, (x_i, y_i) \sim P_S \\
T_m = \{(x_i, y_i), i=1...m\}, x_i \sim P_T\\
$$

And our data distributions are
$$
P_S(x, y) = P_S(y | x) P_S(x) \\
P_T(x, y) = P_S(y | x) P_T(x) \\
$$

For shorthand, we will write $P_S(y | x)$ as $P(y | x)$.

Let $h$ be the classifier. We wish to minimize the risk $R_T(h)$. FOr simplicity, we will assume binary classification:

$$
R_T(h) = E_{(x, y) \sim P_T} |h(x) - y|\\
= \sum_{x, y}P(y | x) P_T(x) |h(x) - y \\
= \sum_{x, y}P(y | x) P_S(x) \frac{P_T(x)}{P_S(x)} |h(x) - y| \\
= E_{(x, y) \sim P_S} \frac{P_T(x)}{P_S(x)} |h(x) - y|\\
\approx \frac{1}{n} \sum_{i=1}^n \frac{\hat{P}_T(x)}{\hat{P}_S(x)}|h(x) - y|
$$

This formulation has several challenges. First, we operate under the assumption that the source distribution covers the target distribution, because otherwise it would lead to infinity. Second, when $x$ is high-dimensional, it is difficult to estimate $\frac{\hat{P}_T(x)}{\hat{P}_S(x)}$.

To address the second issue, we train a mixture model to predict the probability of a point belonging to either the source or target distribution ($Q(S | x)$ and $Q(T | x)$, respectively). Then, we have
$$
\frac{\hat{P}_T(x)}{\hat{P}_S(x)} \propto \frac{Q(T | X)}{Q(S | X)}
$$

## Unsupervised Domain Adaptation Theory
We first define a discrepancy measure between source and target distributions:
$$
R_T(h) = E_{(x, y) \sim P_T} |h(x) - y| \\ 
R_T(h, h') = E_{(x, y) \sim P_T} |h(x) - h'(x)| \\ 
d_{\mathcal{H} \Delta \mathcal{H}} (P_T, P_S) = \sup_{h, h' \in \mathcal{H}} |R_T(h, h') - R_S (h, h')|
$$
Then, it can be proven that
$$
R_T(h) \leq R_S(h) + d_{\mathcal{H} \Delta \mathcal{H}} (P_T, P_S) + \min_{h \in \mathcal{H}} (R_T(h, h') + R_S (h, h'))
$$
Qualitatively, the $d_{\mathcal{H} \Delta \mathcal{H}}$ can be interpreted as how different the performances are across distributions and the $\min$ term indicates how good the performance is in general. This inequality holds even outside the context of covariate shift.

## Domain Adversarial Training
We would like to apply the source classifier to the target examples after applying a feature transformation $z = \phi_w(x)$. We want $\phi_w(x)$ to support classification of source examples and also make source and target domains look alike $P_s(z = \phi_w(x)) \approx P_T(z = \phi_w(x))$.

This can be cast as a regularization problem:
$$
E_{(x, y) \sim P_s} L(f(\phi_w(x), y)) + \lambda d(P_{\phi, S}, P_{\phi, T})
$$
where $d(P_{\phi, S}, P_{\phi, T})$ is a divergence measure (for example, Jensen-Shannon Divergence $\max_Q \left[E_{x \sim P_S} \log Q(S | \phi_w(X)) + E_{x \sim P_T} \log Q(T | \phi_w(X))\right]$).

The overall objective function looks like this:
$$
\min_{\phi_w(x), f} \max_Q E_{(x, y) \sim P_s} L(f(\phi_w(x), y)) + \lambda\left[E_{x \sim P_S} \log Q(S | \phi_w(X)) + E_{x \sim P_T} \log Q(T | \phi_w(X))\right]
$$

# In Summary
- Supervised doman adaptation has many labeled examples for the source task and a few labeled examples for the target task. We can pretrain on the source and finetune on the target.
- Unsupervised domain adaptation has the issue of covariate shift and we need to align the target task with the source. Adversarial domain alignment estimates a feature mapping that allows source examples to be classified well and matches source and target distributions.
- Domain generalization (and out of distribution generalization) is when we have multiple source tasks but an unknown target task that might differ similarly from the source tasks as the source tasks differ from each other.