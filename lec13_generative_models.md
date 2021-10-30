# Generative Models
Presented by Dr. Tommi Jaakkola, MIT, Oct 26 2021

Notes by Emily Liu

## Motivation
Often, we want to be able to generate data of various types (data records, images, text, graphs, etc). Being able to generate data from a distribution has many uses, including filling in missing data and making complex inferences aobut examples, identifying regularities in the data, etc.

Fundamentally, we want to estimate the distribution (density) of the data $P(x; \theta)$ and sample $x$ from this distribution.

## Autoregressive language modeling
We want to be able to model a sequence of words into a sentence. We draw our words $x_i$ from the set of all possible words $V$. Since sentences are of variable length, we include the sentence end delimiter (\<end\>) to show that we do not continue generating words.

We wish to learn a distribution $P(X_1 = x_1, X_2 = x_2 ... X_k =$ \<end\>$)$.

This distribution can be written as $P(x_1)P(x_2 | x_1) ... P(x_k | x_1 ... x_{k-1})$.

We can model this through an RNN where the input to the next iteration of the RNN is the output of the previous ($x_i = f(x_{i-1})$) until the end delimiter is reached. In this manner, we have created a sentence predictor that samples from the distribution of next possible words given all the previous words in the sentence.

The autoregressive method is not limited to language. It can also be used to generate images and graphs.

## Steps toward latent variable models

### K-means clustering
K-means clustering is one form of unsupervised representation learning wherein we select $k$ cluster centers, assign each data point to a cluster center, and redefine cluster centers according to the mean of each point.

We define the objective of K-means clustering as
$$
J(Q_{ij}, \mu_j) = \sum_{i-1}^n \sum{j=1}^k Q_{ij} ||x_i - \mu_j||^2
$$
where $Q_ij \in \{0,1\}$ denotes assignment variables of points to clusters.

We can optimize this objective function via alternating minima.

One issue with K-means is that it doesn't do well with overlapping, unequally distributed, or oddly shaped clusters. Instead, we can try to explicitly define and estimate a generative process for the samples. We can build complex generative models from simpler components (ie: Bernoulli, Categorical, univariate/spherical Gaussians, etc).

# In summary:
- Generative models approximate a distribution over the sample space and generate new examples by sampling from this distribution.
- Autoregressive modeling samples points conditionally based on previous information and can be modeled with a RNN.
- K-means clustering can be adapted into a generative process (mixture model - see next lec).