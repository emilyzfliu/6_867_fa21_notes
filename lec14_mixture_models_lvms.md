# Mixture Models, Latent Variable Models
Presented by Dr. Tommi Jaakkola, MIT, Oct 28 2021

Notes by Emily Liu

## Mixture models
Last time, we covered K-means clustering and defined the goal of replacing the output of the algorithm with a distribution, thereby creating a generative model based on K-means.

A mixture model is a type of model that involves discrete or continuous latent variables. In the context of extending K-means, we can consider having a separate generative model ($P(z)$) for each identified cluster $z$, and can generate points $x$ by sampling from $P(x|z)$. The overall $P(x)$ in this case is given by $\sum_{z=1}^k P(x | z) P(z)$.

## Gaussian Mixture Model

A K-component Gaussian Mixture Model is defined as follows:
$$
P(x; \theta) = \sum_{z=1}^k P(z) P(x|z) = \sum_{z=1}^k \pi_z N(x; \mu_z, \sigma_z)
$$

Observe we take the previous formula and let each individual distribution be its own Gaussian.

## The EM Algorithm

Our goal is to estimate this mixture model from "unlabeled" data $D = \{x_1 ... x_n\}$ by maximizing the log likelihood:

$$
l(D; \theta) = \sum_{i=1}^n \log{P(x_i; \theta)} = \sum_{i=1}^n \log{\left[\sum_{z=1}^k \pi_z N(x; \mu_z, \sigma_z)\right]}
$$

This is hard! So hard, in fact, that the method of solving this exact maximization problem is called the **hard EM algorithm**. The objective function for the hard EM algorithm is given by $J(Q, \theta) = \sum_{i=1}^n \sum{j=1}^k Q_{ij} \log{\pi_j N(x_{ij}; \mu_j, \sigma_j)}$ and can be solved via alternating minima (first maximizing over fixed params $\pi, \mu, \sigma$ for $Q$, then maximizing over optimal $Q$ for the parameters).

### Relaxation
We can relax the EM algorithm (for any mixture model) by thinking of $Q$ as a distribution. We first consider just one sample $x$.

$$
\max \log P(x) = \log\sum_{z=1}^k P(z) P(x | z)\\
= \log{\sum_{z=1}^k Q(z | x) \frac{P(z) P(x | z)}{Q(z | x)}} \geq \sum_{z=1}^k Q(z | x) \log \frac{P(z) P(x | z)}{Q(z | x)}\\
= \sum_{z=1}^k Q(z | x) \log(P(z) P(x | z)) + \sum_{z=1}^k Q(z|x) \log \frac{1}{Q(z|x)}
$$

We can make the greater than or equal to adjustment due to Jensen's inequality.

Note that the first term in the final expression corresponds to expected complete log likelihood and the second term corresponds to Shannon Entropy. We can maximize this the same way as previously (via altmin first for $Q$ and then for $\theta$). The EM algorithm is called this because the first step corresponds to expectation and the second step corresponds to maximization.

## Properties of the EM Algorithm
- For a single point $x$, maximizing over $Q$ leads to $Q^{(0)}(j | x) = P(j | x, \theta^{(0)})$; $\log P(x; \theta^{(0)}) = J(Q^{(0)}, \theta^{(0)})$
- EM iterates lead to a non-decreasing sequence of log likelihoods.

One issue with the mixture model approach is that it is with fixed $k$, meaning that you will need to specify the number of clusters prior to running. However, this isn't always information that you will have.

## Looking ahead: Bayesian Networks
Bayesian networks are directed acyclic graphs wherein a directed edge from $x$ to $y$ suggests that $y$ is conditional on $x$. We can use these graphical networks to describe how complex probability distributions factor into smaller components.

# In summary
- Gaussian mixture models (and mixture models in general) turn K-means clustering into a generative algorithm by having each cluster center be approximated by a distribution.
- Mixture models are described by the maximum log likelihood objective function, which is maximized via the EM algorithm.
- Bayesian networks abstract complex conditional relations between random variables through use of a DAG.