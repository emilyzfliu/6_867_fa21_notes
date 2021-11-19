# Conditional GANs, Flow & Diffusion Models
Presented by Dr. Pulkit Agarwal, MIT, Nov 16 2021

Notes by Emily Liu

## Motivation
Much of current deep learning is performed according to an objective function designed by a human (MSE, cross entropy, VLB, etc). However, the issue is that these human-designed loss functions may not be ideal for the task. For example, image encoding/decoding tasks can achieve low loss but still produce images that lack detail when sampled from the latent space. The problem is not that the network is not performing well; it is that the loss function fails to capture the level of detail necessary. Thus introduces the need to be able to design a good objective function.

## GANs in context
Recall that a generative adversarial network consists of a generator $G$ and a discriminator $D$ optimized according to the expression $\argmin_G \max_D E_{x, y}\left[\log D(G(x)) + \log(1 - D(y))\right]$. From the perspective of $G$, $D$ is a loss function because it changes according to how "badly" $G$ is performing. Instead of being hand-designed, $D$ is learned from the data and is highly structured.

To emphasize that we compare the difference between generator output to the true input, we will also include the input $x$ into our generator, making the objective function

$$
\argmin_G \max_D E_{x, y}\left[\log D(x, G(x)) + \log(1 - D(x, y))\right]
$$

## Conditional GAN
The optimal generator $G^*$ for the conditional GAN is given by
$$
G^* = \argmin_G \max_D L_{cGAN}(G, D) + \lambda L_{L1}(G)
$$

where $L_cGAN$ is the loss described in the previous section and $L_{L1}$ is the L1 norm.

Conditional GANs (and other structured objective tasks) have been shown to outperform deep learning with unstructured objectives (such as least squares regression).

## Flow Models
Flow models are another type of generative model in which we attempt to learn both the mapping of $x$ to some encoding $z$ **and** the inverse (decoding) of $z$ back to $x$. More formally, given $x \sim \hat{p}(x)$ and $z \sim \hat{p}(z)$, we have $z = f(x)$ and $x = f^{-1}(z)$.

For example, let $p(z) = N(0,1)$ the standard normal. Then, based on the model parameters $\theta$, $p_{theta}(x) = \int_{z} p_{\theta}(x | z) p(z) dz$. However, this is intractable.

There is an easier way to compute the relationship between $p(x)$ and $p(z)$. Assuming $f_\theta$ is a bijection, we can state that $p(x) dx = p(z) dz$, so $p(x) = p(z)\frac{dz}{dx} = p(z) |J|$ where $|J|$ is the Jacobian.

We want to maximize the Jacobian while still maintaining both computational efficiency (as determinant calculation is cubic time) and expressivity (as making the matrix too simple nerfs the complexity of the mapping we would like to make).

To achieve this, we couple the layers. We split $z$ into $z_1$ and $z_2$ and $x$ into $x_1$ and $x_2$. Then, we let $z_1 = x_1$ and $z_2 = s_\theta(x_1) \bigodot x_2 + t_\theta(x_1)$. As a result, the Jacobian matrix becomes an upper triangular matrix (quick to compute determinant and also expressive).

The issue with this setup is that half the image is modeled well and the other half is modeled poorly. Therefore, we will need to perform this operation twice, switching $x_1$ and $x_2$.

If we apply multiple of these mappings $f$, we get the inverse is $|f_1^{-1} \bigodot f_2^{-1} ... f_N^{-1}|$. Because this is called **normalizing flow**, this model is called a flow model.

Although flow models are not as crisp as GANs, they are useful in that they can provide us the exact likelihoods, while GANs are only able to provide lower bounds.

## Diffusion Models
The logic behind the diffusion model is that if we gradually add small bits of noise to an input image, we will eventually get an isotropic Gaussian. If we are able to reverse these steps (using a neural network), then we can theoretically go from an isotropic Gaussian back to a real image.

To formalize, the image at each state is given by $x_t$. The noise at each state is given by $q(x_t | x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$, where $\beta$ gives the diffusion rate (low $\beta$ is low noise and vice versa). What we want to compute is the probability of working backwards, $p_\theta(x_t | x_{t-1})$.

We can express all $q$ in terms of $x_0$:

$$
q_(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})
$$

Then, we reparametrize using $\alpha_t = 1 - \beta_t$ to get
$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} z_{t-1} \\
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} z
$$
where $z$ is standard normal and $\bar{\alpha} = \prod_{i=1}^T \alpha_i$.

To reverse this, we want to calculate $q(x_{t-1} | x_t)$:
$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta (x_{t-1} | x_t) \\
p_\theta (x_{t-1} | x_t) = N (x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$
where $\mu_\theta(x_t, t), \Sigma_\theta(x_t, t)$ are parameters learned using the neural network.

Note: because we know the $q$ for everything when we know $x_0$, the reverse process $q(x_{t-1} | x_t, x_0)$ is tractable (we know mean and variance at each step). However, when we don't, we use the neural network to predict mean and variance. It can be shown that the relation between these is equivalent to variational lower bound.

# In Summary
- Conditional GANs treat the discriminator as a structured learnable loss function that better fits the image generation task.
- Flow models learn forward encodings and their inverse decodings at the same time; layer coupling is used to ensure the Jacobian is upper triangular and therefore efficient to compute.
- Diffusion models add gradual amounts of noise to an image until it is an isotropic Gaussian and attempts to learn the reverse transformation.
