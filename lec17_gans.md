# Deep Generative Models, Variational Autoencoders
Presented by Dr. Pulkit Agarwal, MIT, Nov 9 2021

Notes by Emily Liu

## Motivation
Machine learning tasks fall into two broad categories: analysis and synthesis. In analysis, you want to go from a data to a label (representation learning). In synthesis, you want to go from a label to a data point (generative modelling).

## Image synthesis with GANs
The generative adversarial network (GAN) consists of two parts: a generator and a discriminator. The generator samples points from a prior distribution ($z$, often standard normal) and transforms it into a data sample. The discriminator is a neural net that tries to identify whether a sample is artificially generated or a ground truth sample.

The total loss of a GAN is given by

$$
\min_G \max_D E_{z, x} \left[\log D(G(z)) + \log (1 - D(x))\right]
$$

and is minimized at $D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{gen}(x)}$ where $p_{data}(x)$ is the probability $x$ came from the original dataset and $p_{gen}(x)$ is the probability $x$ came from the generator. The minimal value of loss is given when $p_{gen} = p_{data}$.

## GANs vs VAEs
GANs and VAEs are very similar because they can both be used for generative modeling. However, VAEs have the issue of blurry samples when randomly sampling points from the latent space, and GANs have the issue of worse coverage than VAEs and being more difficult to train.

## Types of GANs
### Self-Attention GANs
Self-Attention GANs use attentions instead of convolution in the model. Attention layers have the benefit of modeling long-range dependencies in the generator and enforcing complicated constraints on global image structure in the discriminator. (It is harder to do these with convolutions as convolutions mainly model local structure, which may not be the most important when trying to generate an image close to the real thing).

The SAGAN loss function is given by
$$
L_D = -E_{(x, y) \sim p_{data}}[\min (0, D(x, y) - 1)] - E_{z \sim p_z, y \sim p_{data}}[\min (0, -D(G(z), y) - 1)] \\
L_G = - E_{z \sim p_z, y \sim p_{data}}[D(G(z), y)]
$$

### BigGAN
BigGAN is an optimization on the standard GAN. The main differences are
- Larger batch size (2048)
- More parameters (2x increase in both G and D)
- Orthogonal Weight Initialization
- Truncation trick (capping the magnitude of values sampled from prior)
- Orthogonality Regularization $R_\beta(W) = \beta ||W^T W _ I||_F^2$
- Optimizing D twice as often as G

### Style GANs
Style GANs add a new branch called the style branch that learns the **style** (affine transformation) and **noise** (channel-wide normalization) at each layer. The model aligns the mean and variance of the content features with the style features through the use of adaptive instance normalization $AdaIN(x_i, y) = y_{s, i}\frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b, i}$. The generative model can select the style from one image and the content from another.

## Improving GANs

### Spectral and Gradient Normalization

One issue when training GANs is the singular values will grow rapidly resulting in collapse. One way to address this issue is through spectral normalization, where we cap a maximum desired value of the first singular value $\sigma_0$. The update rule is $W = W - \max(0, \sigma_0 - \sigma_{clamp})v_0u_0^T$ where $v_0$ and $u_0$ are the right and left eigenvectors. (The singular valuse can be found via the power iteration method.)

Another way to prevent this explosion is through graient normalization:

$$
R_i := \frac{\gamma}{2} E_{p_D(x)} [||\nabla D(x)||_F^2]
$$

where high $\gamma$ stabilizes training but worsens performance.

### Mode Collapse Problem
The mode collapse problem occurs when the generator learns a few "good" distributions that will pass the discriminator and ends up generating very similar images, resulting in a lack of diversity. This can be solved by increasing the batch size.

### Data Support Issue
When the priors for the natural data distribution and the generated image distribution have no overlap, it is impossible to optimize the functions as the KL divergence of the two distributions will be infinite. Therefore, we use a different metric, the Wasserstein Distance $W(P, Q) = |\theta|$. Using the Wasserstein Distance, we can define a Wasserstein Metric
$$
W(p_r, p_s) = \inf_{\gamma \sim \Pi(p_r, p_s)} E_{(x, y) \sim \gamma} [||x - y||]
$$

The WGAN (Wasserstein GAN) can therefore avoid the issue of exploding/vanishing gradients.

## Measuring GAN Performance
Currently, there doesn't exist a great way to measure GAN performance. Ideally, we want the GAN to generate images that are realistic, diverse, and in agreement with human evaluation. Additionally, we prefer models with disentangled feature space, sensitivity to image transformations, and low sample and computational complexity.

Two currently existing GAN metrics are the Inception Score (IS) and the Frechet Inception Distance (FID).

The Inception Score is given by $\exp\left(E_x(KL(p(y|x) || p(y)))\right) = \exp\left(H(y) - E_x[H(y | x)]\right)$ where $p(y | x)$ is the probability of getting $y$ given $x$ from the Inception network trained on Imagenet.

The Frechet Inception Distance compares the mean and covariance of features from the real and generated samples on the second to last layer of the Inception net:
$$
FID(r, g) = ||\mu_r - \mu_g||^2_2 + Tr\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

# In Summary
- GANs are a class of generative model that make use of a generator net (samples from prior and turns to image/data point) and a discriminator (determine how realistic/good generated data point is)
- Using self-attention in a GAN can improve performance by modeling global dependencies in the image
- Style GANs learn both style and content of an image and can mix and match them in their generative modeling
- Some issues that GANs run into are collapse/singular value explosion (solvable by spectral/gradient normalization), mode collapse (solvable by larger batch size), data support issue (solvable by WGANs)
- Inception Score and FID can be used to evaluate GANs but there exists no universal metric for evaluating GAN performance