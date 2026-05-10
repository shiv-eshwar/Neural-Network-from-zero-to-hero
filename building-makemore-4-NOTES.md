# Neural Networks: Zero to Hero

## Lecture 4 - Building Makemore, Part 3: Activations, Gradients, Initialization, BatchNorm

> **Instructor:** Andrej Karpathy  
> **Main goal:** Understand what is happening *inside* a neural network while it trains: activations, gradients, initialization, saturated nonlinearities, dead neurons, Kaiming initialization, batch normalization, and diagnostic plots.

This lecture is less about making the model architecture more powerful and more about learning how to **debug and stabilize neural network training**.

If Lecture 1 taught us how backprop works, Lecture 2 taught us how language-model loss works, and Lecture 3 taught us the MLP character model, this lecture teaches:

> How to look inside a neural network and tell whether it is in a healthy state.

---

## Table of Contents

1. [Why This Lecture Matters](#1-why-this-lecture-matters)
2. [Starting Point: The Previous MLP Model](#2-starting-point-the-previous-mlp-model)
3. [`torch.no_grad()` During Evaluation](#3-torchnograd-during-evaluation)
4. [Problem 1: Initial Loss Is Way Too High](#4-problem-1-initial-loss-is-way-too-high)
5. [Fixing the Output Layer Initialization](#5-fixing-the-output-layer-initialization)
6. [Why Not Set All Weights to Exactly Zero?](#6-why-not-set-all-weights-to-exactly-zero)
7. [Problem 2: Saturated `tanh` Activations](#7-problem-2-saturated-tanh-activations)
8. [Backprop Through `tanh`: Why Saturation Kills Gradients](#8-backprop-through-tanh-why-saturation-kills-gradients)
9. [Dead Neurons](#9-dead-neurons)
10. [Fixing the Saturated Hidden Layer](#10-fixing-the-saturated-hidden-layer)
11. [Kaiming Initialization: A More Principled Scale](#11-kaiming-initialization-a-more-principled-scale)
12. [Why Initialization Used to Be So Fragile](#12-why-initialization-used-to-be-so-fragile)
13. [Batch Normalization: The Core Idea](#13-batch-normalization-the-core-idea)
14. [BatchNorm Gain and Bias](#14-batchnorm-gain-and-bias)
15. [BatchNorm During Inference: Running Mean and Variance](#15-batchnorm-during-inference-running-mean-and-variance)
16. [Why BatchNorm Bias Makes Previous Bias Useless](#16-why-batchnorm-bias-makes-previous-bias-useless)
17. [BatchNorm Summary](#17-batchnorm-summary)
18. [Real-World Pattern: Linear/Conv -> BatchNorm -> Nonlinearity](#18-real-world-pattern-linearconv---batchnorm---nonlinearity)
19. [PyTorch-ifying the Code With Modules](#19-pytorch-ifying-the-code-with-modules)
20. [Custom `Linear`, `BatchNorm1d`, and `Tanh` Layers](#20-custom-linear-batchnorm1d-and-tanh-layers)
21. [Visual Diagnostic 1: Forward Activations](#21-visual-diagnostic-1-forward-activations)
22. [Visual Diagnostic 2: Backward Gradients](#22-visual-diagnostic-2-backward-gradients)
23. [The Fully Linear Case: Why Nonlinearities Matter](#23-the-fully-linear-case-why-nonlinearities-matter)
24. [Visual Diagnostic 3: Parameter and Gradient Statistics](#24-visual-diagnostic-3-parameter-and-gradient-statistics)
25. [Visual Diagnostic 4: Update-to-Data Ratio](#25-visual-diagnostic-4-update-to-data-ratio)
26. [Bringing Back BatchNorm in a Deeper MLP](#26-bringing-back-batchnorm-in-a-deeper-mlp)
27. [What BatchNorm Fixes, and What It Does Not Fix](#27-what-batchnorm-fixes-and-what-it-does-not-fix)
28. [Final Code Skeleton](#28-final-code-skeleton)
29. [Final Cheat Sheet](#29-final-cheat-sheet)

---

## 1. Why This Lecture Matters

We want to move from the MLP model toward more complex networks like:

- recurrent neural networks,
- GRUs,
- LSTMs,
- eventually Transformers.

But before doing that, Karpathy pauses at the MLP level because we need a strong intuition for:

- activations during the forward pass,
- gradients during the backward pass,
- why bad initialization can ruin training,
- why nonlinearities can block gradient flow,
- why normalization layers were invented,
- how to inspect a network and see if it is healthy.

### Plain English Intuition

Think of a neural network as a long chain of water pipes.

- The **forward pass** sends information through the pipes.
- The **backward pass** sends learning signals back through the pipes.
- If one pipe explodes, the signals become too large.
- If one pipe pinches shut, the signals vanish.
- If many pipes are badly sized, training becomes unstable or impossible.

This lecture teaches how to check whether the pipes are sized properly.

### Technical Translation

We want activation distributions and gradient distributions that are neither:

- exploding to huge values, nor
- shrinking to zero.

For deep networks, this is critical. A shallow MLP may still train even if badly initialized. A deep network can completely fail.

---

## 2. Starting Point: The Previous MLP Model

The starting point is the MLP from the previous lecture:

```text
context characters
    -> embedding table C
    -> concatenate embeddings
    -> hidden linear layer
    -> tanh
    -> output linear layer
    -> logits
    -> cross entropy loss
```

The previous good-ish configuration was:

```python
block_size = 3
n_embd = 10
n_hidden = 200
```

The model has about:

```text
11,000 parameters
```

Training for about 200,000 steps gave around:

```text
train loss ~= 2.16
val loss   ~= 2.17
```

Karpathy refactors the notebook:

- removes magic numbers,
- makes `n_embd` and `n_hidden` explicit,
- adds a cleaner split evaluation function,
- keeps the same sampling code,
- prepares to inspect the internals.

---

## 3. `torch.no_grad()` During Evaluation

During training, PyTorch tracks operations so it can later do:

```python
loss.backward()
```

This tracking builds a computation graph.

During evaluation, we do not need gradients.

So Karpathy writes evaluation functions using:

```python
@torch.no_grad()
def split_loss(split):
    ...
```

### Plain English

When you are just checking the score, you do not need to remember every calculation for a future backward pass.

So `torch.no_grad()` tells PyTorch:

> "Do not build the training memory graph here. I promise I will not call backward."

### Why It Matters

This makes evaluation:

- faster,
- more memory efficient,
- cleaner.

You can also use it as a context manager:

```python
with torch.no_grad():
    ...
```

---

## 4. Problem 1: Initial Loss Is Way Too High

At the very beginning of training, the model records:

```text
initial loss ~= 27
```

This is a red flag.

### What Loss Should We Expect at Initialization?

At initialization, before learning anything, the model should not strongly prefer any character.

There are 27 possible next characters.

If the model predicts uniformly:

```text
probability of correct character = 1 / 27
```

Cross entropy loss is negative log probability:

```python
-torch.tensor(1/27.0).log()
```

which is:

```text
3.2958
```

So the initial loss should be around:

```text
3.29
```

not:

```text
27
```

### Plain English

At the start, the model should say:

> "I have no idea. Every character is equally likely."

That is not great, but it is honest.

Instead, the model is saying:

> "I am extremely sure this wrong character is next."

Being confidently wrong creates a huge loss.

### Technical Reason

The output logits are too extreme at initialization.

Softmax turns logits into probabilities. If one wrong logit is very large, softmax gives it almost all the probability mass. The true label gets tiny probability, and:

```text
-log(tiny probability) = huge loss
```

---

## 5. Fixing the Output Layer Initialization

The logits are computed by:

```python
logits = h @ W2 + b2
```

To make logits near zero at initialization:

1. Set output bias `b2` to zero.
2. Make output weights `W2` small.

Instead of:

```python
W2 = torch.randn((n_hidden, vocab_size))
b2 = torch.randn(vocab_size)
```

use:

```python
W2 = torch.randn((n_hidden, vocab_size)) * 0.01
b2 = torch.randn(vocab_size) * 0
```

Now initial logits are small and close to zero.

The initial loss becomes close to:

```text
3.29
```

which is what we expect.

### Why This Improves Final Training

Before the fix, the first part of training was wasted:

- The optimizer first had to squash huge logits down.
- Only after that could it learn useful structure.

After the fix:

- The model starts in a sane state.
- Training steps are spent learning useful relationships.

This also removes the "hockey stick" shape in the loss curve, where the loss initially drops sharply only because the network is reducing overconfidence.

---

## 6. Why Not Set All Weights to Exactly Zero?

For the output layer specifically, setting `W2 = 0` might be fine.

But in general, we do **not** initialize all neural-network weights to zero.

### Plain English

If every neuron starts with the exact same weights, then every neuron behaves exactly the same. They receive the same gradients and continue to learn the same thing.

That means 200 neurons might behave like only 1 neuron copied 200 times.

### Technical Name

This is called a failure of:

```text
symmetry breaking
```

Randomness in initialization gives different neurons different starting points, allowing them to specialize.

So we prefer:

```text
small random numbers
```

not:

```text
exact zeros
```

---

## 7. Problem 2: Saturated `tanh` Activations

After fixing output logits, there is still a deeper problem.

The hidden activations are:

```python
hpreact = embcat @ W1 + b1
h = torch.tanh(hpreact)
```

When Karpathy plots a histogram of `h`, many values are exactly or almost:

```text
-1 or +1
```

That means `tanh` is saturated.

### What `tanh` Does

`tanh` squashes any real number into:

```text
[-1, 1]
```

When input is near zero, the function is steep and sensitive.

When input is very positive or very negative, the function becomes flat:

```text
large positive input -> tanh ~= +1
large negative input -> tanh ~= -1
```

### Plain English

If a neuron is in the flat part of `tanh`, changing its input barely changes its output.

If changing the input does not change the output, then the loss also does not change through that path.

So backprop says:

> "This parameter has no useful effect."

and sends almost no gradient.

---

## 8. Backprop Through `tanh`: Why Saturation Kills Gradients

From micrograd, we know:

```python
def _backward():
    self.grad += (1 - t**2) * out.grad
```

where:

```text
t = tanh(x)
```

So:

```text
d/dx tanh(x) = 1 - tanh(x)^2
```

If `t = 1`:

```text
1 - 1^2 = 0
```

If `t = -1`:

```text
1 - (-1)^2 = 0
```

So when `tanh` saturates at `+1` or `-1`, the gradient becomes zero.

### Plain English

The neuron becomes like a stuck switch.

No matter how much you try to adjust its input, the output stays basically the same, so learning cannot flow through it.

### Technical Consequence

Saturated `tanh` units cause:

```text
vanishing gradients
```

This becomes much worse in deep networks, because every saturated layer multiplies gradients by small numbers.

---

## 9. Dead Neurons

Karpathy visualizes saturation using:

```python
plt.imshow(h.abs() > 0.99, cmap='gray')
```

White means:

```text
this activation is saturated
```

Rows are examples. Columns are hidden neurons.

### Dead `tanh` Neuron

If an entire column is white, then that hidden neuron is saturated for every example.

That means:

```text
no example gives that neuron useful gradient
```

This is a dead neuron.

### Dead ReLU Neuron

For ReLU:

```python
relu(x) = max(0, x)
```

The negative side is perfectly flat.

If a ReLU neuron always receives negative preactivations, it always outputs zero and receives zero gradient.

Then it can die permanently.

### How Neurons Die

Dead neurons can happen:

- at initialization,
- during training,
- because the learning rate is too large,
- because a neuron gets pushed far away from the data manifold.

Karpathy calls this:

```text
permanent brain damage in the network
```

### Nonlinearities Mentioned

| Nonlinearity | Problem |
|---|---|
| `tanh` | saturates near `-1` and `+1` |
| sigmoid | saturates near `0` and `1` |
| ReLU | dead if always negative |
| leaky ReLU | less likely to die because negative side still has slope |
| ELU | can still have flat-ish regions |

---

## 10. Fixing the Saturated Hidden Layer

The preactivations are:

```python
hpreact = embcat @ W1 + b1
```

They are too large and spread out. So `tanh(hpreact)` saturates.

To fix this:

1. Make `b1` small or zero.
2. Scale down `W1`.

Example:

```python
W1 = torch.randn((n_embd * block_size, n_hidden)) * 0.2
b1 = torch.randn(n_hidden) * 0.01
```

Now `hpreact` values are closer to zero.

Then `tanh` activations are less saturated.

The model improves:

| Fix | Validation Loss |
|---|---:|
| previous model | about 2.17 |
| fixed output layer overconfidence | about 2.13 |
| fixed saturated `tanh` hidden layer | about 2.10 |

### Big Lesson

Bad initialization wastes training time and can damage gradient flow.

Good initialization puts the network into a healthy regime immediately.

---

## 11. Kaiming Initialization: A More Principled Scale

So far, we guessed numbers like:

```python
0.2
0.01
```

But we do not want magic numbers. We want a principled way to choose weight scale.

### The Core Problem

Suppose:

```python
x = torch.randn(1000, 10)
W = torch.randn(10, 200)
y = x @ W
```

If `x` has standard deviation 1 and `W` has standard deviation 1, then `y` has a larger standard deviation, around 3.

So the layer expands the distribution.

If repeated over many layers, activations can explode.

### Goal

We want each layer to preserve roughly the same activation scale:

```text
input std ~= output std
```

### Fan-In

The number of inputs to a neuron is called:

```text
fan_in
```

For a matrix:

```python
W.shape == (fan_in, fan_out)
```

the fan-in is the first dimension.

### Standard Deviation Preserving Scale

For a linear layer, a good scale is:

```text
1 / sqrt(fan_in)
```

So:

```python
W = torch.randn(fan_in, fan_out) / fan_in**0.5
```

### With Nonlinearities: Gain

Nonlinearities squash distributions.

So we multiply by an extra factor called:

```text
gain
```

For `tanh`, PyTorch recommends:

```text
gain = 5 / 3
```

For ReLU, Kaiming/He initialization uses:

```text
gain = sqrt(2)
```

because ReLU zeroes about half the distribution.

### Kaiming-Style Initialization for This Model

Here:

```python
fan_in = n_embd * block_size
```

For `n_embd = 10` and `block_size = 3`:

```text
fan_in = 30
```

For `tanh`:

```python
W1 = torch.randn((fan_in, n_hidden)) * (5/3) / (fan_in**0.5)
```

This gives a scale around:

```text
0.304
```

which is close to the manually guessed value.

### PyTorch Support

PyTorch includes:

```python
torch.nn.init.kaiming_normal_
```

and related initialization utilities.

The key ideas are:

- choose a scale based on fan-in,
- include a gain based on the nonlinearity,
- keep activations and gradients roughly controlled.

---

## 12. Why Initialization Used to Be So Fragile

Before modern deep learning tricks, training deep networks was extremely delicate.

You had to carefully control:

- weight scale,
- activation variance,
- gradient variance,
- nonlinearity saturation,
- optimizer behavior.

If you got this wrong, deep networks often did not train.

Modern techniques make this less fragile:

- residual connections,
- batch normalization,
- layer normalization,
- group normalization,
- better optimizers like RMSProp and Adam.

But the underlying issues still matter.

This lecture teaches the "old-school" diagnostic intuition so you understand what these modern tools are fixing.

---

## 13. Batch Normalization: The Core Idea

Batch normalization was introduced in 2015 and had a huge impact because it made deep networks much easier to train.

### Problem

We want hidden preactivations to be roughly:

```text
mean = 0
standard deviation = 1
```

at least at initialization, so nonlinearities do not saturate.

### BatchNorm Insight

Instead of carefully initializing everything so activations *happen* to be Gaussian-like, why not directly normalize them?

Given:

```python
hpreact.shape == (batch_size, n_hidden)
```

Compute mean over the batch:

```python
batch_mean = hpreact.mean(0, keepdim=True)
```

Compute standard deviation over the batch:

```python
batch_std = hpreact.std(0, keepdim=True)
```

Normalize:

```python
hpreact_norm = (hpreact - batch_mean) / batch_std
```

Now each hidden neuron has roughly:

```text
mean = 0
std  = 1
```

over the current batch.

### Plain English

If a layer's outputs are too messy, BatchNorm says:

> "Let's clean them up before passing them to the next layer."

It recenters and rescales every hidden unit using the statistics of the current minibatch.

### Why This Is Allowed

Mean, variance, subtraction, division are all differentiable operations.

So BatchNorm can be inserted into the network and trained with backpropagation.

---

## 14. BatchNorm Gain and Bias

If we force every hidden unit to always have mean 0 and standard deviation 1, we may restrict the network too much.

Sometimes the network may want:

- wider activations,
- narrower activations,
- shifted activations,
- more trigger-happy or less trigger-happy neurons.

So BatchNorm includes learnable scale and shift parameters:

```python
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
```

Then:

```python
hpreact = bngain * (hpreact - batch_mean) / batch_std + bnbias
```

### Plain English

BatchNorm first normalizes the activations, then gives the network two knobs:

- `bngain`: "How wide should this neuron's values be?"
- `bnbias`: "Where should this neuron's values be centered?"

At initialization:

```text
bngain = 1
bnbias = 0
```

so it starts as pure normalization.

During training, backpropagation can change them.

### Parameters

BatchNorm trainable parameters:

- gain, often called `gamma`,
- bias, often called `beta`.

These are optimized using gradient descent.

---

## 15. BatchNorm During Inference: Running Mean and Variance

BatchNorm creates a strange issue.

During training, it uses the current batch mean and standard deviation.

But during inference, we may want to pass **one example** through the network.

What batch statistics do we use then?

### Stage 2 Calibration Option

After training, run the entire training set through the network once and compute:

```python
bnmean = hpreact.mean(0, keepdim=True)
bnstd = hpreact.std(0, keepdim=True)
```

Then during inference use those fixed statistics:

```python
hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias
```

This works, but people do not like an extra calibration stage.

### Running Statistics During Training

Instead, maintain running estimates during training:

```python
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))
```

At each training step:

```python
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * batch_mean
    bnstd_running = 0.999 * bnstd_running + 0.001 * batch_std
```

These are **not** trained by backprop. They are buffers updated on the side.

### PyTorch Version

PyTorch BatchNorm stores:

- `running_mean`,
- `running_var`,
- trainable `weight` (gamma),
- trainable `bias` (beta).

At training time:

- use batch statistics,
- update running statistics.

At eval/inference time:

- use running statistics,
- do not update them.

---

## 16. Why BatchNorm Bias Makes Previous Bias Useless

Suppose:

```python
hpreact = embcat @ W1 + b1
```

Then BatchNorm subtracts the batch mean:

```python
hpreact - hpreact.mean(0)
```

Any constant bias `b1` added before BatchNorm is subtracted away by the mean.

So `b1` becomes useless.

### Practical Rule

If a linear or convolution layer is immediately followed by BatchNorm, set:

```python
bias=False
```

or do not include that bias at all.

BatchNorm already has its own bias parameter:

```python
bnbias
```

### Real PyTorch Pattern

This is why many ResNet blocks have:

```python
nn.Conv2d(..., bias=False)
nn.BatchNorm2d(...)
nn.ReLU()
```

The convolution bias would be redundant.

---

## 17. BatchNorm Summary

BatchNorm has:

### Trainable Parameters

These are updated by backpropagation:

```text
gamma / gain
beta / bias
```

### Buffers

These are updated manually, not by backprop:

```text
running mean
running variance
```

### Training Forward Pass

```python
mean = x.mean(0, keepdim=True)
var = x.var(0, keepdim=True)
xhat = (x - mean) / torch.sqrt(var + eps)
out = gamma * xhat + beta
```

Also update running statistics.

### Inference Forward Pass

```python
xhat = (x - running_mean) / torch.sqrt(running_var + eps)
out = gamma * xhat + beta
```

### Why `eps` Exists

BatchNorm divides by standard deviation.

If variance becomes zero, division would blow up.

So BatchNorm uses:

```python
eps = 1e-5
```

and divides by:

```python
sqrt(var + eps)
```

This avoids division by zero.

---

## 18. Real-World Pattern: Linear/Conv -> BatchNorm -> Nonlinearity

Karpathy shows a ResNet example.

A common deep learning block is:

```text
weight layer -> normalization layer -> nonlinearity
```

For MLPs:

```text
Linear -> BatchNorm -> Tanh/ReLU
```

For CNNs:

```text
Conv2d -> BatchNorm2d -> ReLU
```

### Why Convolutions Are Similar to Linear Layers

A convolution is basically a linear layer applied to local image patches.

Instead of multiplying the entire input vector by `W`, it applies a learned filter across spatial patches.

But conceptually:

```text
convolution ~= structured linear operation
```

So the same pattern applies:

```text
Conv/Linear -> BatchNorm -> Nonlinearity
```

### Why `bias=False` in ResNet Convs?

Because BatchNorm follows immediately after.

The BatchNorm layer subtracts the mean, removing the effect of the previous bias, and then applies its own bias.

So the convolution bias is redundant.

---

## 19. PyTorch-ifying the Code With Modules

In the bonus section, Karpathy rewrites the code to look more like PyTorch.

Instead of manually writing:

```python
h = torch.tanh(x @ W + b)
```

he defines layer classes:

- `Linear`
- `BatchNorm1d`
- `Tanh`

Then stacks them:

```python
layers = [
    Linear(n_embd * block_size, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
]
```

This looks like PyTorch's:

```python
nn.Sequential(...)
```

### Why Do This?

It helps separate:

- layer initialization,
- layer forward pass,
- parameters of each layer.

This makes the code modular and closer to production style.

---

## 20. Custom `Linear`, `BatchNorm1d`, and `Tanh` Layers

### Custom `Linear`

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
```

This mirrors `torch.nn.Linear`.

### Custom `BatchNorm1d`

Important pieces:

- `gamma`: trainable scale,
- `beta`: trainable shift,
- `running_mean`: buffer,
- `running_var`: buffer,
- `training`: flag controlling train vs eval behavior.

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * xmean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * xvar
                )

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

### Custom `Tanh`

```python
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
```

This layer has no parameters.

### `.out` Attribute

Karpathy adds:

```python
self.out
```

to each layer so he can later inspect activations and gradients.

This is not how PyTorch modules normally work, but it is useful for teaching and diagnostics.

---

## 21. Visual Diagnostic 1: Forward Activations

Karpathy plots histograms of activations for each `Tanh` layer.

He computes:

- mean,
- standard deviation,
- saturation percentage.

Saturation percentage:

```python
(t.abs() > 0.97).float().mean() * 100
```

### What We Want

For `tanh` activations:

- mean near 0,
- standard deviation not collapsing to 0,
- not too many saturated values near -1 or +1,
- similar distributions across layers.

### If Gain Is Too Small

Activations shrink toward zero layer by layer.

Plain English:

> The signal fades as it moves through the network.

### If Gain Is Too Large

Activations become saturated near -1 and +1.

Plain English:

> The signal smashes into the walls of `tanh`, and gradients die.

### Good Gain for `tanh`

PyTorch suggests:

```text
gain = 5/3
```

Empirically, this keeps activation histograms stable in a stack of Linear -> Tanh layers.

---

## 22. Visual Diagnostic 2: Backward Gradients

Karpathy also plots histograms of gradients flowing through each `Tanh` output:

```python
t = layer.out.grad
```

We want:

- gradients that are not shrinking to zero,
- gradients that are not exploding,
- roughly similar gradient distributions across layers.

### Why This Matters

The backward pass multiplies gradients by many local derivatives.

If each layer shrinks gradients slightly, then 50 layers can almost erase the gradient.

If each layer expands gradients slightly, then 50 layers can explode the gradient.

This is one reason deep networks were hard to train before modern initialization and normalization techniques.

---

## 23. The Fully Linear Case: Why Nonlinearities Matter

Karpathy briefly removes `tanh` layers and makes the network a stack of only linear layers.

### Mathematical Fact

A stack of linear layers is still just one linear layer.

Example:

```text
y = W3(W2(W1x + b1) + b2) + b3
```

can be collapsed into:

```text
y = Wx + b
```

So without nonlinearities, a deep network does not gain representational power.

### Plain English

Stacking many straight-line transformations still gives one straight-line transformation.

To model complicated curves and decisions, we need nonlinearities.

### Technical Note

Even though the forward function collapses to a single linear layer, the optimization dynamics can differ across deep linear stacks. There are papers studying this. But for representation power, nonlinearities are essential.

---

## 24. Visual Diagnostic 3: Parameter and Gradient Statistics

Karpathy inspects the weights themselves and their gradients.

For each weight matrix:

- shape,
- mean,
- standard deviation,
- gradient standard deviation,
- gradient-to-data ratio.

### Gradient-to-Data Ratio

```text
std(grad) / std(data)
```

This tells us how large the gradients are compared to the parameter values.

If gradients are huge compared with weights, updates may be unstable.

If gradients are tiny compared with weights, learning may be too slow.

### Output Layer Oddity

The final layer can look different because we intentionally scaled it down:

```python
last_layer.weight *= 0.1
```

to make softmax less confident at initialization.

Because its data values are artificially small, its gradient-to-data ratio can appear larger at the beginning. This often stabilizes after training for a while.

---

## 25. Visual Diagnostic 4: Update-to-Data Ratio

Gradient-to-data is useful, but the actual update is:

```python
update = learning_rate * gradient
```

So a better diagnostic is:

```text
std(update) / std(data)
```

Karpathy tracks:

```python
ud.append([
    ((lr * p.grad).std() / p.data.std()).log10().item()
    for p in parameters
])
```

### Rule of Thumb

He likes update-to-data ratio to be around:

```text
10^-3
```

On a log10 plot:

```text
-3
```

### Interpretation

If ratio is much higher, like `10^-1`:

```text
updates are too aggressive
learning rate may be too high
```

If ratio is much lower, like `10^-5`:

```text
updates are too tiny
learning rate may be too low
```

### Plain English

Imagine your parameter is a knob.

If each training step twists it by 50%, training is probably unstable.

If each step twists it by 0.000001%, training is probably too slow.

Around 0.1% per step is often a reasonable rough scale.

---

## 26. Bringing Back BatchNorm in a Deeper MLP

Karpathy then inserts BatchNorm layers between Linear and Tanh:

```python
Linear -> BatchNorm1d -> Tanh
Linear -> BatchNorm1d -> Tanh
Linear -> BatchNorm1d -> Tanh
...
```

Now activation histograms look much more stable automatically.

### What BatchNorm Helps With

BatchNorm makes the forward activations well-behaved even when the weight scale is not perfectly calibrated.

It also helps backward gradients become better behaved.

### But It Is Not Magic

Even with BatchNorm:

- update-to-data ratios can still be off,
- learning rate may need retuning,
- parameter update scales can change,
- batch coupling introduces noise and complexity.

### BatchNorm and Gain

Without BatchNorm, gain must be carefully chosen.

With BatchNorm, forward activation distributions are more robust to gain changes because BatchNorm explicitly normalizes them.

But the weight gradients and update ratios can still change, so the learning rate may need adjustment.

---

## 27. What BatchNorm Fixes, and What It Does Not Fix

### BatchNorm Fixes / Helps

- activation scale drift,
- saturation at initialization,
- training instability in deep nets,
- some gradient-flow issues,
- dependence on perfect initialization.

### BatchNorm Does Not Fully Fix

- bad learning rate,
- poor architecture,
- too-short context length,
- insufficient model capacity,
- noisy minibatch statistics,
- the weird coupling between examples in the same batch.

### Why BatchNorm Is Weird

Before BatchNorm, examples in a batch were processed independently. Batching was just an efficiency trick.

With BatchNorm, the output for one example depends on the other examples in the batch, because the batch mean and variance are shared.

So if the same example appears in different batches, its hidden activations can slightly change.

This creates noise.

Surprisingly, that noise can act as a regularizer, making overfitting harder.

But it also introduces bugs and complexity.

Karpathy says people often dislike BatchNorm for this reason and prefer alternatives like:

- layer normalization,
- group normalization,
- instance normalization.

Still, BatchNorm was historically very important because it made deep networks much easier to train.

---

## 28. Final Code Skeleton

Below is a simplified code skeleton capturing the lecture's final ideas.

### Setup

```python
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
block_size = 3
```

### Dataset

```python
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
```

### Layers

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * xmean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * xvar
                )

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
```

### Model

```python
n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
    Linear(n_embd * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
]

with torch.no_grad():
    # less confident final layer
    layers[-1].weight *= 0.1

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True
```

### Training

```python
max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

    emb = C[Xtr[ix]]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ytr[ix])

    for layer in layers:
        layer.out.retain_grad()

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 100000 else 0.01
    with torch.no_grad():
        ud.append([
            ((lr * p.grad).std() / p.data.std()).log10().item()
            for p in parameters
            if p.ndim == 2
        ])

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())
```

### Evaluation

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'dev': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]

    emb = C[x]
    out = emb.view(emb.shape[0], -1)
    for layer in layers:
        out = layer(out)
    loss = F.cross_entropy(out, y)
    print(split, loss.item())

for layer in layers:
    layer.training = False

split_loss('train')
split_loss('dev')
```

---

## 29. Final Cheat Sheet

### Expected Initial Loss

For `vocab_size = 27`:

```python
-log(1/27) ~= 3.29
```

If initial loss is much larger, the model is probably confidently wrong.

### Healthy Output Layer Initialization

```python
W2 *= 0.01
b2 *= 0
```

Goal:

```text
initial logits near zero
initial softmax near uniform
```

### `tanh` Derivative

```text
d/dx tanh(x) = 1 - tanh(x)^2
```

If `tanh(x)` is near `+1` or `-1`, gradient is near zero.

### Saturation Check

```python
(h.abs() > 0.99).float().mean()
```

High saturation means many neurons are in flat `tanh` regions.

### Dead Neuron

A neuron is dead if it never receives useful gradient over the dataset.

For `tanh`:

```text
always saturated at -1 or +1
```

For ReLU:

```text
always inactive, output 0
```

### Kaiming / Fan-In Initialization

For a linear layer:

```python
W = torch.randn(fan_in, fan_out) / fan_in**0.5
```

With nonlinearity:

```python
W = torch.randn(fan_in, fan_out) * gain / fan_in**0.5
```

Common gains:

| Nonlinearity | Gain |
|---|---:|
| linear | 1 |
| tanh | 5/3 |
| ReLU | sqrt(2) |

### BatchNorm Formula

Training:

```python
mean = x.mean(0, keepdim=True)
var = x.var(0, keepdim=True)
xhat = (x - mean) / torch.sqrt(var + eps)
out = gamma * xhat + beta
```

Inference:

```python
xhat = (x - running_mean) / torch.sqrt(running_var + eps)
out = gamma * xhat + beta
```

### BatchNorm Parameters vs Buffers

| Type | Examples | Updated By |
|---|---|---|
| parameters | `gamma`, `beta` | backpropagation |
| buffers | `running_mean`, `running_var` | exponential moving average |

### Bias Rule

If using:

```text
Linear/Conv -> BatchNorm
```

then usually:

```python
bias=False
```

because BatchNorm subtracts the mean and has its own bias.

### Diagnostic Plots to Inspect

| Diagnostic | What to Look For |
|---|---|
| activation histograms | not saturated, not collapsed |
| gradient histograms | not exploding, not vanishing |
| parameter histograms | sane scales |
| grad-to-data ratio | gradients not wildly bigger/smaller than weights |
| update-to-data ratio | around `10^-3` as rough heuristic |

### Update-to-Data Ratio

```python
((lr * p.grad).std() / p.data.std()).log10()
```

Good rough region:

```text
around -3
```

Too high:

```text
updates too large, learning rate may be too high
```

Too low:

```text
updates too small, learning rate may be too low
```

### Three Sentences That Summarise This Lecture

1. Neural networks train well only when activations and gradients stay in healthy ranges; bad initialization can make logits overconfident, saturate nonlinearities, and kill gradients.
2. Kaiming initialization and gain scaling give a principled way to set weight scales, while BatchNorm directly normalizes activations and makes deep networks much easier to train.
3. A serious practitioner watches activation histograms, gradient histograms, parameter statistics, and update-to-data ratios to diagnose whether the network is learning in a healthy way.

---

## Beginner-Friendly One-Page Summary

If you are new to neural networks, remember this:

Neural networks learn by sending numbers forward and learning signals backward. If the forward numbers become too huge or too tiny, or if the backward learning signals become too huge or too tiny, training becomes bad.

This lecture shows three common problems:

1. The final layer can start too confident, causing a huge initial loss.
2. `tanh` neurons can saturate at `-1` or `+1`, causing gradients to vanish.
3. Deep networks can slowly explode or shrink signals as they pass through many layers.

The fixes are:

1. Make the final layer small so the initial predictions are uniform.
2. Initialize hidden layers so activations stay in a good range.
3. Use normalization layers like BatchNorm to keep activations controlled.

The deep learning engineer's job is not just to write a model, but also to check whether the model is internally healthy.

---

## Technical One-Page Summary

The lecture continues the makemore MLP and audits its initialization and gradient flow.

Key technical points:

- Expected initial cross entropy for 27 classes is `-log(1/27) = 3.2958`.
- Initial loss of 27 indicates overly extreme logits and confident wrong predictions.
- Fix final logits by scaling output weights down and zeroing output bias.
- `tanh` saturation kills gradients via local derivative `1 - t^2`.
- Hidden preactivations should not be too broad; scale `W1` appropriately.
- Kaiming-style initialization uses `std = gain / sqrt(fan_in)`.
- For `tanh`, PyTorch's recommended gain is `5/3`.
- BatchNorm normalizes preactivations over the batch, then applies learned `gamma` and `beta`.
- BatchNorm running stats are buffers, not gradient-trained parameters.
- Linear/Conv layers followed by BatchNorm usually do not need bias.
- Diagnostic plots of activations, gradients, weights, and update-to-data ratios reveal miscalibration.
- A rough update-to-data target is `10^-3`.

---

> **End of notes for Lecture 4 - Building Makemore, Part 3: Activations, Gradients, Initialization, BatchNorm.**
>
> The core lesson: before building bigger networks, learn to inspect whether your current network is numerically healthy.
