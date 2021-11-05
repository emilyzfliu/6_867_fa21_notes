# Latent Variable Models, Variational Learning

## Recall: Bayesian Networks
Bayesian networks use directed acyclic graphs to describe how the probability distribution factors into smaller components. A directed edge from node $u$ to $x$ signals that the distribution of $x$ is conditional on $u$ (so the corresponding probability distributions correspond to $P(u)$ and $P(x | u)$ respectively).

## Bayesian Networks with Plates
Some graph networks will have boxes subscripted with a constant (let's say $N$) drawn around groups of nodes. This means that there were $N$ iid samples drawn from the distribution described by the subgraph in the box.

Bayesian networks can be applied to describe many problems. A few are listed below.

## Bayesian matrix factorization
Consider a $m \times n$ matrix where the entry at row $i$ with features $u_i$ and column $j$ with features $v_j$ is given by $x_{ij}$, conditioned on $u_i$ and $v_j$. The probability distribution is given by

$$
P(u, v, x) = \left[\prod_{i=1}^n P(u_i)\right]\left[\prod_{j=1}^m P(v_j)\right]\left[\prod_{i=1}^n \prod_{j=1}^m P(x_{ij} | u_i, v_j)\right]
$$

To simplify, we consider $u_i$ and $v_j$ to be both $d$ dimensional vectors, and $P(u_i)$ and $P(v_j)$ to be normal distributions with mean $\mu$ and variance $w^2 I$. Let $P(x_{ij} | u_i, v_j)$ be a normal with mean $u_i^T v_j$ and variance $\sigma^2$.

In our matrix, our known values form our prediction dataset $D := \{x_{kl} \forall k, l \in D\}$. Our prediction

$$
P(x_{ij} | D) = \int_{u_i} \int_{v_j} P(x_{ij} | u_i, v_j) P(u_i, v_j | D) dv_j du_i\\
= \int_{u_i} \int_{v_j} N(x_{ij}; u_i^T v_j, \sigma^2) P(u_i, v_j | D) dv_j du_i
$$

The distribution $P(u_i, v_j | D)$ is the term that is hardest to compute in the above expression, so you need an algorithm to learn it.s

In this formulation, you would only need to train $d+2$ parameters: d parameters for $\mu$, and 1 parameter for $\sigma^2$ and $w^2$ each.

## Multi-task clustering

Sometimes, clusters remain the same across tasks, but the proportion of examples in each cluster changes. We need to model the variability of mixing proportions across the tasks,
$$
P(x) = \sum_{z=1}^k \pi_k N(x; \mu_k, \Sigma_k)
$$

The probability distribution can be constructed as follows:

For $t = 1...T$:
- $\pi_t \sim$ Dir($\alpha_1, ..., \alpha_k$) 
- For $i = 1...N$:
  - $z_{it} \sim$ Cat($\pi_{t1}...\pi_{tk}$)
  - $x_{it} \sim N(\mu_{z_{it}}, \Sigma_{z_{it}})$

Where Dir is the Dirichlet distribution.

## LDA Topic Model

Similar analysis can be applied to topic models where you draw from a categorical distribution instead of a normal distribution. (In this case, $z$ corresponds to a topic selected from the clusters, and $w$ corresponds to a word from the selected topic)

For $i = 1...N$:
- $\pi_t \sim$ Dir($\alpha_1, ..., \alpha_k$) 
- For $i = 1...N$:
  - $z_{i} \sim$ Cat($\theta_1...\theta_k$)
  - $w_{i} \sim$ Cat($\{\beta_{w | z_i}\}_{w \in W}$)

And the probability of all the words in a single doc being
$$
P(w1...w_N) = \int P(\theta | \alpha) \prod_{i=1}^N\left(\sum_{z_i=1}^k \theta_z \beta_{w_i | z_i}\right)d \theta
$$

## LDA, EM, ELBO
Consider in the context of LDA topic identification a single document $d = w1...w_N$. The loss is given by
$$
l(d, \alpha, \beta) = \log \int P(\theta | \alpha) \prod_{i=1}^N\left(\sum_{z_i=1}^k \theta_z \beta_{w_i | z_i}\right)d \theta
$$ 

In order to use EM to estimate $\alpha, \beta$, we need to evaluate and maximize the variational lower bound.

We do this via expected lower bound optimization (ELBO)

$$
l(d, \alpha, \beta) \geq \sum_{z_1...z_n} \int Q(\theta, z_1 ... z_N) \log P(\theta | \alpha) \prod_{i=1}^N\theta_z \beta_{w_i | z_i} d\theta + H(Q) = ELBO(Q, \alpha, \beta)
$$

This is not tractable. $Q(\theta, z_1 ... z_N)$  is proportional to the posterior $P(\theta | \alpha) \prod_{i=1}^N \theta_{z_i} \beta_{w_i | z_i}$, which is itself not tractable. However, we can use a technique called **mean field approximation** to approximate $Q$ and perform EM optimization.

# In summary
- Bayesian networks describe complex probability distributions
- Plates in Bayesian networks describe repeated iid sampling from the same probability distribution
- Bayesian networks can be used to model matrix completion by treating the features of the columns and rows as distributions. However, this is still a difficult task.
- Bayesian networks can be used to model multi-task clustering and topic models in a similar fashion.
- The variational lower bound is maximized via ELBO through the EM algorithm; however to make this feasible we need to approximate $Q$ via mean field approximation.