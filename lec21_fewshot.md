# Few-Shot and Lifelong Learning
Presented by Dr. Pulkit Agarwal, MIT, Nov 23 2021

Notes by Emily Liu

## Motivation
The general motivation for the class of transfer and few-shot learning tasks is that after pretraining a model, we should be able to perform a new unseen task faster, or perform an unseen task that's more complex than the training tasks.

Previously, we covered multitask and transfer learning. In multitask learning, we learn N iid tasks simultaneously with the reasoning that many features will be shared between them. In transfer learning, we have a domain shift in data, meaning we finetune our pretrained task to fit our new domain.

In few-shot learning, we update model parameters at test time, often to create new object categories. As the name suggests, this is (intuitively) doable with fewer data points than is required for training from scratch (in practice, about 50-100s of labelled finetuning points).

## More on finetuning
When finetuning a network there is a question of which layers we should update. Empericially, it has been shown that when there is less data, finetuning the last two layers is sufficient, but with more data, we can finetune all the layers.

## Similarity learning
Suppose we have a two-class few-shot learning problem where our training set consists of one sample from each class ($(x_1, y_1), (x_2, y_2)$). Our test set is one sample belonging to one of the two classes and we need to figure out which.

One strategy is to use the neural network to compute embeddings for the training points ($z_1$, $z_2$) and the test point $z$ and select the class whose embedding has the smaller distance from $z$. The issue with this formulation is that there's no guarantee that the neural network preserves similarity.

### Siamese Networks
One solution to this problem is to evaluate pairs of similar or dissimilar samples on the same neural network. If the samples are similar, the cosine distance between their embeddings should be close to 1, and if the samples are different, the cosine distance between their embeddings should be close to 0.

### Matching Networks
Concepts from the Siamese network can be extended to multiple classes using the matching network. Matching networks take a support set $S$ of images and pass through a pretrained net. The new sample $\hat{x}$ is passed through a finetuned network, and attention values $a$ are computed as the softmax of cosine distances $c(a, b)$:
$$
a(\hat{x}, x_i) = \frac{\exp(c(g_\theta(x_i), f(\hat{x})))}{\sum_{i=1}^k \exp(c(g_\theta(x_i), f(\hat{x})))}
$$

The likelihood of belonging to class $y$ is given by $P(y | \hat{x}, S) = \sum_{i}^k a(\hat{x}, x_i), y_i$.

Additionally, it has been noted that the useful features of the support set can depend on what images we want to classify. Therefore, performance can be improved further with contextual embeddings that change per task.

## Another perspective on transfer learning
We can think of the pretrained parameters $\theta$ as a vector in high dimensional space, and finetuned parameters $\theta_1$, $\theta_2$ as other vectors in this same space. The amount we need to train can be given roughly as $\delta \theta_1 = |\theta_1 - \theta|$ and $\delta \theta_2 = |\theta_2 - \theta|$. We want to minimize $\delta \theta_1 + \delta \theta_2$, which we can achieve by setting $\theta$ at the midpoint between $\theta_1$ and $\theta_2$.

## Using Large Models

### Contrastive Pretraining
The task is to match images with captions. We train the images on an image encoder and the captions on a text encoder to capture similar information. Then take a cosine distance between the image and text encodings. The higher cosine distances are more likely to be a relevant image-caption match.

### Zero-Shot Prediction
Given a new (uncaptioned) image, we pass it through the image encoder. Then, we find a match through the text encodings to "generate" a caption for the image without any extra training.

## Catastrophic Forgetting
In sequential learning tasks, where we take a pretrained set of parameters and continually finetune on new tasks, we run into the issue of catastrophic forgetting, which means that the network trained on new tasks is unable to perform the old tasks.

To deal with catastrophic forgetting, we can remember the weights from each task and accept features from the previous networks as additional inputs into each layer of the new networks (ProgressiveNet).

# In Summary
- Few-shot learning is finetuning on few unseen examples. In practice, few-shot learning can be implemented via Siamese networks or matching networks.
- Contrastive pretraining/zero shot learning leverages big data to "generate" labels for unseen test images.
- Catastrophic forgetting occurs when a finetuned network is unable to "remember" old tasks. This is avoided by feeding previous weights into new networks.