# Deep Generative Models, Variational Autoencoders
Presented by Dr. Tommi Jaakkola, MIT, Nov 4 2021

Notes by Emily Liu

## Motivation
Generative models are useful because they allow us to generate diverse, high quality objects (images, time series, sentences, molecules, etc).

To do this, we estimate a latent variable model $P(x | z; \theta)P(z)$ from the data and generate new examples by sampling $z \sim P(z)$, $x \sim P(x | z; \hat{\theta})$.

## ELBO
Consider a single value of $x$.

$$
\log P(x | \theta) = \log \left[ \int P(x | z, \theta) dz \right] \geq E_{z \sim Q_{z|x}} \{\log \left[P(x | z, \theta) P(z)\right]\} + H(Q_{\cdot | x})
$$

If we run the EM algorithm with no constraints on Q, the ELBO max results in $\hat{Q}(z | x) = P(z | x, \theta)$.

However, $P(z | x, \theta)$ is difficult to compute, so we need to simplify $Q$ with the mean field approximation. Namely, we assume $Q$ factorizes across the samples.
$$
\hat{Q}(z | x) = \prod_i \hat{Q_i}(z_i | x)
$$
We then maximize ELBO over restricted $Q$s separately for each new observation $x$.

## ELBO Mean Field Updates
Let us consider a simple case with two latent variables: $P(x, z_1, z_2)$ where by the mean field approximation we get $Q(z_1, z_2) = Q_1(z_1) Q_2(z_2)$ that best approximates $P(z_1, z_2 | x)$ for each $x$.

We optimize iteratively (first fix $Q_1(z_1)$ and optimize over $Q_2(z_2)$, then vice versa until convergence).

$$
ELBO = \sum_{z_1} \sum_{z_2} Q_1(z_1) Q_2(z_2) \log P(x, z_1, z_2) + H(Q_1) + H(Q_2) \\
= \sum_{z_2} Q_2(z_2) \left[\sum_{z_1} Q_1(x_1) \log(P(x, z_1, z_2)) \right] + H(Q_2) + \text{const} \\
= \sum_{z_2} Q_2(z_2) \left[E_{z_1 \sim Q_1} \log(P(x, z_1, z_2)) \right] + H(Q_2) + \text{const}
$$

The $E_{z_1 \sim Q_1} \log(P(x, z_1, z_2))$ is the effective, or "mean" log likelihood.

Solving, we get that $\hat{Q_2}(z_2)$ is proportional to $\exp(E_{z_1 \sim Q_1} \log P(x, z_1, z_2))$.

## Variational Autoencoders
Variational autoencoders utilize a parametric model $Q(z | x, \phi)$ that typically factors as the mean field approximation.

Generative models generate new objects as follows:
- First, sample $z \sim P(z)$ from a simple fixed distribution.
- Then, map the resulting $z$ value through a deep model $x = g(z; \theta)$
- The generated objects tend to be noisy; in other words, $P(x | z, \theta) = N(x; g(z; \theta); \sigma^2 I)$

This model is difficult to train since we don't know which $z$ corresponds to which $x$ (because $z$ is unobserved). We can use variational autoencoders to infer the $z$ associated with $z$ using an encoder network. The encoder network predicts the mean $\mu(x; \phi)$ and standard deviation $\sigma(x; \phi)$ of $z$. The encoder and decoder networks need to be learned together.

The ELBO objective is as follows:

$$
\log \left[\int P(x | z, \theta) P(z) dz\right] \geq \int Q(x | z, \phi) \log(P(x|z, \theta) P(z)) dz + H(Q) \\ 
= \int Q(x | z, \phi) \log(P(x|z, \theta)) dz + KL(Q_{z | x; \phi} || P_z) = ELBO(Q_\phi; \theta)
$$

## VAE Update Steps
1. Given $x$, pass $x$ through the encoder to get $\mu(x; \phi)$ and $\sigma(x; \phi)$ that specify $Q(z|x, \phi)$
2. Sample $\epsilon \sim N(0, I)$ so that $z_{\phi} = \mu(x; \phi)+\sigma(x; \phi) \odot \epsilon$ corresponds to a sample from the encoder distribution $Q(z|x, \phi)$
3. Update the decoder using this $z_{\phi}$:
$$
\theta = \theta + \eta \nabla_\theta \log P (x | z_\phi, \theta)
$$
4. Update encoder to match the stochastic ELBO criterion:
$$
\phi = \phi + \eta \nabla_\theta \{\log P(x | z_\phi, \theta) - KL(Q_{z|x; \phi} || P_z)\}
$$

# In summary
- Deep generative models realize objects from latent randomization by approximating the probability space $P(x | z) P(z)$ and drawing samples $z$ and $x$ from the probability space.
- If we had the latent random vector $z$ together with the image as pairs, optimization of the architecure would be easy as it would become a supervised learning task. However, since we don't have the pair, we resort to EM
- EM is not tractable for these complex distribution problems so we need an encoder network to infer latent $z$ variable (output of E step)
- VAEs use an encoder network to infer $z$ from $x$ and a decoder network to map $x$ back to $z$. The decoder is also the generator network
- VAE architecture is learned by maximizing a lower bound on the log likelihood of the data via the ELBO criterion.