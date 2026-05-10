# Neural Networks: Zero to Hero

## Lecture 3 - Building Makemore, Part 2: MLP Character-Level Language Model

> **Instructor:** Andrej Karpathy  
> **Main idea:** Move beyond the bigram model by using a Bengio-style multi-layer perceptron language model. Instead of predicting the next character from only one previous character, we predict it from a fixed-size context window of previous characters.

---

## Table of Contents

1. [Why the Bigram Model Breaks Down](#1-why-the-bigram-model-breaks-down)
2. [Bengio et al. 2003: The MLP Language Model](#2-bengio-et-al-2003-the-mlp-language-model)
3. [Building the Training Dataset](#3-building-the-training-dataset)
4. [Embedding Lookup Table `C`](#4-embedding-lookup-table-c)
5. [Embedding Many Contexts at Once](#5-embedding-many-contexts-at-once)
6. [Hidden Layer: Flattening Embeddings and Using `view`](#6-hidden-layer-flattening-embeddings-and-using-view)
7. [Output Layer and Logits](#7-output-layer-and-logits)
8. [Negative Log-Likelihood Loss](#8-negative-log-likelihood-loss)
9. [`F.cross_entropy`: Why Use the Built-In](#9-fcross_entropy-why-use-the-built-in)
10. [Training Loop and Overfitting One Batch](#10-training-loop-and-overfitting-one-batch)
11. [Training on the Full Dataset With Minibatches](#11-training-on-the-full-dataset-with-minibatches)
12. [Finding a Good Learning Rate](#12-finding-a-good-learning-rate)
13. [Learning Rate Decay](#13-learning-rate-decay)
14. [Train / Dev / Test Splits](#14-train--dev--test-splits)
15. [Underfitting, Overfitting, and Scaling the Network](#15-underfitting-overfitting-and-scaling-the-network)
16. [Visualising 2-D Character Embeddings](#16-visualising-2-d-character-embeddings)
17. [Increasing Embedding Size](#17-increasing-embedding-size)
18. [Sampling From the MLP Model](#18-sampling-from-the-mlp-model)
19. [Final Code Skeleton](#19-final-code-skeleton)
20. [Final Cheat Sheet](#20-final-cheat-sheet)

---

## 1. Why the Bigram Model Breaks Down

In the previous lecture, we built a **bigram character-level language model**. It predicted the next character using only the **single previous character**.

That model can be represented as a table:

- Rows: current character.
- Columns: next character.
- Each row: a probability distribution over the next character.

For one character of context, the table has:

```text
27 rows x 27 columns
```

where 27 = 26 lowercase letters + the special `.` token.

This works, but the samples are not very good because the model has almost no context. It knows only "what character came immediately before."

### The Exponential Blow-Up

If we try to extend the table to more context, the number of possible contexts explodes:

| Context Length | Number of Contexts |
|---:|---:|
| 1 previous character | 27 |
| 2 previous characters | 27 x 27 = 729 |
| 3 previous characters | 27^3 = 19,683 |
| 10 previous characters | 27^10, far too many |

The table grows **exponentially** with context length. Most rows would have very few counts, many would never appear, and the model would not generalize.

So we move to a **neural network** that can take multiple previous characters as input without needing an explicit count table for every possible context.

---

## 2. Bengio et al. 2003: The MLP Language Model

Karpathy follows the modeling approach from:

> **Bengio et al. (2003), "A Neural Probabilistic Language Model"**

The paper is word-level, but this lecture adapts it to characters.

### Paper Setup

In the paper:

- Vocabulary size: about 17,000 words.
- Each word gets embedded into a lower-dimensional vector, e.g. 30 dimensions.
- The model takes several previous words and predicts the next word.

In our lecture:

- Vocabulary size: 27 characters.
- Each character gets embedded into a lower-dimensional vector, initially 2 dimensions.
- The model takes several previous characters and predicts the next character.

### Why Embeddings Help

The central idea is to learn a **distributed representation**:

- Similar words/characters can be placed near each other in embedding space.
- Knowledge can transfer across similar contexts.

Example from the paper:

```text
A dog was running in a ___
```

Maybe this exact phrase never appeared in training. But the model may have seen similar phrases:

```text
The dog was running in a ___
A cat was walking in a ___
```

If embeddings learn that:

- `a` and `the` are similar,
- `dog` and `cat` are similar,
- `running` and `walking` are similar,

then the model can generalize to unseen combinations.

### Architecture in the Paper

For a context of 3 words:

```text
word indices
    -> embedding lookup table C
    -> concatenate embeddings
    -> fully connected hidden layer
    -> tanh non-linearity
    -> output layer
    -> softmax over next-word vocabulary
```

The parameters are:

- Embedding table `C`.
- Hidden layer weights and biases.
- Output layer weights and biases.

All are trained with backpropagation by maximizing the log-likelihood of the training data, or equivalently minimizing negative log-likelihood.

---

## 3. Building the Training Dataset

We rebuild the dataset, but this time every example consists of:

- `X`: a fixed-length context of previous characters.
- `Y`: the next character after that context.

### Vocabulary and Mappings

```python
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
```

The `.` token means both "start" and "end", and has index `0`.

### Context Length: `block_size`

```python
block_size = 3
```

This means:

> Use the previous 3 characters to predict the next one.

So for a name like `emma`, padded with start/end dots, examples look like:

```text
... -> e
..e -> m
.em -> m
emm -> a
mma -> .
```

### Dataset Construction

```python
block_size = 3
X, Y = [], []

for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
```

Key idea:

- `context` is a rolling window.
- It starts as all zeros: `[0, 0, 0]`.
- Each time we see a new character, we append the current context to `X` and the next character to `Y`.
- Then we shift the context left and append the new character.

For the full dataset, this creates about **228,000 examples**.

---

## 4. Embedding Lookup Table `C`

Instead of one-hot encoding characters and multiplying by a weight matrix, we directly create an **embedding lookup table**.

```python
C = torch.randn((27, 2))
```

Shape:

```text
C.shape == (27, 2)
```

Meaning:

- 27 rows: one row per character/token.
- 2 columns: each character is represented by a 2-D vector.

Initially these vectors are random. During training, backpropagation will move them around so that useful characters are placed in useful parts of the embedding space.

### Embedding a Single Character

```python
C[5]
```

This retrieves the embedding vector for character index `5`.

### Connection to One-Hot Encoding

In the previous lecture, we encoded an integer as a one-hot vector, then multiplied by a matrix.

For index `5`:

```python
F.one_hot(torch.tensor(5), num_classes=27).float() @ C
```

This gives the exact same result as:

```python
C[5]
```

Why? The one-hot vector is all zeros except one `1`, so matrix multiplication masks out every row except row `5`.

So:

```text
one_hot(index) @ C == C[index]
```

Indexing is much faster and cleaner, so from now on we use direct embedding lookup.

---

## 5. Embedding Many Contexts at Once

PyTorch indexing is powerful. You can index a matrix not just with one integer, but with an entire tensor of integers.

If:

```python
X.shape == (32, 3)
C.shape == (27, 2)
```

then:

```python
emb = C[X]
```

has shape:

```text
emb.shape == (32, 3, 2)
```

Interpretation:

- 32 examples.
- 3 context characters per example.
- 2 embedding dimensions per character.

Each integer inside `X` has been replaced by its row in `C`.

Example:

```python
emb[13, 2] == C[X[13, 2]]
```

This confirms that PyTorch applies the lookup elementwise over the integer tensor.

> **Important mental model:** `C[X]` replaces every integer in `X` by its embedding vector.

---

## 6. Hidden Layer: Flattening Embeddings and Using `view`

The hidden layer expects one flat vector per example.

Currently:

```text
emb.shape == (batch_size, block_size, embedding_dim)
```

For the initial setup:

```text
emb.shape == (32, 3, 2)
```

But the hidden layer wants:

```text
(32, 6)
```

because `3 context chars x 2 embedding dimensions = 6 inputs`.

### First Attempt: `torch.cat`

We could concatenate the three embeddings manually:

```python
torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], dim=1)
```

This gives shape `(32, 6)`, but it is ugly and does not generalize if `block_size` changes.

We can use:

```python
torch.cat(torch.unbind(emb, dim=1), dim=1)
```

This generalizes to any `block_size`, but still creates a new tensor and copies memory.

### Better: `view`

```python
emb.view(32, 6)
```

or better:

```python
emb.view(emb.shape[0], 6)
```

or even:

```python
emb.view(-1, 6)
```

The `-1` tells PyTorch:

> Infer this dimension from the number of elements.

### Why `view` Is Efficient

PyTorch tensors have underlying **storage**, which is just a 1-D block of memory. A tensor's shape is a view over that storage.

Calling `.view(...)` usually does **not** copy memory. It only changes metadata such as:

- shape,
- stride,
- storage offset.

So `.view` is much more efficient than concatenation when it works.

### Hidden Layer Parameters

```python
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
```

Forward pass:

```python
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
```

Shape:

```text
h.shape == (32, 100)
```

Meaning:

- 32 examples in the batch.
- 100 hidden activations per example.

### Broadcasting Biases

`emb.view(-1, 6) @ W1` has shape `(32, 100)`.

`b1` has shape `(100,)`.

PyTorch broadcasts `b1` as if it were `(1, 100)` and adds it to every row. This is exactly what we want: the same bias vector added to every example.

Always check broadcasting carefully.

---

## 7. Output Layer and Logits

The output layer maps hidden activations to logits for all 27 possible next characters.

```python
W2 = torch.randn((100, 27))
b2 = torch.randn(27)

logits = h @ W2 + b2
```

Shape:

```text
logits.shape == (32, 27)
```

Meaning:

- One row per example.
- 27 logits per row.
- Each logit corresponds to one possible next character.

These logits are raw real numbers. They are not probabilities yet.

---

## 8. Negative Log-Likelihood Loss

The manual softmax route is:

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
loss = -probs[torch.arange(32), Y].log().mean()
```

Interpretation:

1. `logits.exp()` converts logits to positive fake counts.
2. Dividing by row sums converts fake counts to probabilities.
3. `probs[torch.arange(32), Y]` selects the probability assigned to the correct next character for each example.
4. `.log()` takes log-probabilities.
5. Negative mean gives average negative log-likelihood.

The model wants this loss to be as low as possible.

If the model assigns high probability to the correct next character, the loss goes down.

---

## 9. `F.cross_entropy`: Why Use the Built-In

Instead of manually writing:

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
loss = -probs[torch.arange(batch_size), Y].log().mean()
```

we use:

```python
loss = F.cross_entropy(logits, Y)
```

This computes the same mathematical objective, but is much better in practice.

### Reason 1: Efficiency

The manual version creates intermediate tensors:

- `counts`
- `probs`
- selected probabilities
- log-probabilities

`F.cross_entropy` can use fused kernels and avoid unnecessary intermediate memory.

### Reason 2: More Efficient Backward Pass

Just like in micrograd, a grouped operation can have a much simpler derivative than the individual operations expanded out.

Example from micrograd:

- Forward formula for `tanh` is complicated.
- Backward formula is simple: `1 - tanh(x)^2`.

Similarly, cross-entropy has a simplified backward pass.

### Reason 3: Numerical Stability

Manual softmax can overflow:

```python
logits = torch.tensor([-2, 3, -3, 0, 100])
counts = logits.exp()  # exp(100) can overflow
```

Very positive logits can produce `inf`, then probabilities can become `nan`.

Softmax is invariant to adding/subtracting a constant:

```text
softmax(logits) == softmax(logits - logits.max())
```

PyTorch internally subtracts the maximum logit before exponentiating, so the largest shifted logit becomes `0`, and all others are non-positive. This avoids overflow.

So in practice:

```python
loss = F.cross_entropy(logits, Y)
```

is the right thing to use.

---

## 10. Training Loop and Overfitting One Batch

Before training on all data, Karpathy first overfits a tiny batch: the 32 examples from the first five words.

### Parameter List

```python
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
```

### Training Loop

```python
for _ in range(1000):
    # forward pass
    emb = C[X]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data += -0.1 * p.grad
```

### Why the Loss Drops Very Low

We have:

- about 3,400 parameters,
- only 32 examples.

So the network can almost memorize the batch. This is called **overfitting one batch**.

This is useful as a sanity check:

> If a model cannot overfit a tiny batch, something is probably wrong.

### Why the Loss Cannot Become Exactly Zero

Some contexts map to multiple possible next characters.

For example:

```text
... -> e
... -> o
... -> i
... -> s
```

The exact same input context can have several different labels in the training data. A deterministic model cannot assign probability 1 to all of them at once, so the loss cannot reach exactly zero.

---

## 11. Training on the Full Dataset With Minibatches

Now we remove the "first five words only" restriction and use all names.

Full dataset:

```text
about 228,000 examples
```

Doing a full forward/backward pass over all 228k examples every iteration is slow.

### Minibatch Training

Instead, randomly sample a small subset each step:

```python
ix = torch.randint(0, X.shape[0], (32,))
```

Then train only on those rows:

```python
emb = C[X[ix]]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y[ix])
```

### Why Minibatches Work

The minibatch gradient is only an estimate of the true full-dataset gradient.

But it is usually good enough, and it is much cheaper. In practice:

> It is better to take many cheap approximate-gradient steps than a few expensive exact-gradient steps.

The gradient is noisier, but training becomes much faster.

---

## 12. Finding a Good Learning Rate

The learning rate controls update size:

```python
p.data += -lr * p.grad
```

If `lr` is too small:

- loss barely decreases,
- training is slow.

If `lr` is too large:

- loss becomes unstable,
- training can explode.

### Learning Rate Sweep

Karpathy searches over learning rates exponentially from `0.001` to `1`.

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre
```

This creates:

```text
10^-3, ..., 10^0
0.001, ..., 1
```

During training:

```python
lr = lrs[i]
```

Track:

```python
lri.append(lre[i])
lossi.append(loss.item())
```

Then plot:

```python
plt.plot(lri, lossi)
```

The best initial learning rate is near the valley before the curve becomes unstable.

In this lecture, a good initial value is:

```python
lr = 0.1
```

because `10^-1 = 0.1`.

---

## 13. Learning Rate Decay

After training for a while with `lr = 0.1`, the loss starts plateauing.

At that point, lower the learning rate:

```python
lr = 0.01
```

This is **learning rate decay**.

Intuition:

- Early training: use larger steps to move quickly.
- Late training: use smaller steps to refine the solution.

In the lecture, training with this rough schedule improves the model beyond the bigram loss.

Bigram loss from previous lecture:

```text
about 2.45
```

MLP with context length 3 gets down to roughly:

```text
about 2.3 initially, later lower after tuning
```

---

## 14. Train / Dev / Test Splits

Karpathy warns that a lower training loss does not automatically mean a better model.

A large neural network can memorize the training set. Then:

- training loss becomes very low,
- generated samples may just reproduce training names,
- performance on new unseen names can be poor.

So we split the data:

| Split | Typical Size | Purpose |
|---|---:|---|
| Training | 80% | Optimize parameters using gradient descent |
| Dev / Validation | 10% | Tune hyperparameters |
| Test | 10% | Final evaluation, used sparingly |

### Why Use the Test Set Sparingly?

Every time you inspect test performance and change something based on it, you are implicitly fitting to the test set.

So:

- Use training data to learn weights.
- Use dev data to tune architecture and hyperparameters.
- Use test data only near the end to report final performance.

### Dataset Builder Function

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
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y
```

### Split Words

```python
import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
```

Now train only on `Xtr, Ytr`, evaluate often on `Xdev, Ydev`, and reserve `Xte, Yte` for the final score.

---

## 15. Underfitting, Overfitting, and Scaling the Network

After introducing train/dev splits, Karpathy compares training loss and dev loss.

### If Training Loss ~= Dev Loss

This usually means:

```text
The model is underfitting.
```

The model is not powerful enough to memorize training data, and both training and dev performance are similarly limited.

In this lecture, the small model had similar train/dev losses around `2.3`, so it was underfitting.

### Scaling Hidden Layer Size

Increase hidden neurons:

```python
# from 100 hidden neurons
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
W2 = torch.randn((100, 27))

# to 300 hidden neurons
W1 = torch.randn((6, 300))
b1 = torch.randn(300)
W2 = torch.randn((300, 27))
```

Parameter count rises from about:

```text
3,400 -> 10,000+
```

But simply increasing hidden size does not always immediately improve performance. It may require:

- longer training,
- better learning rate,
- larger batch size,
- better optimization schedule,
- larger embeddings.

---

## 16. Visualising 2-D Character Embeddings

With embedding dimension 2, we can directly plot each character in 2-D space.

```python
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
             ha="center", va="center", color="white")
plt.grid("minor")
```

After training, the embeddings are no longer random.

Observed structure:

- Vowels like `a, e, i, o, u` cluster together.
- The special `.` token is separated from ordinary letters.
- `q` often becomes an outlier because it has unusual behavior.
- Some consonants cluster together.

This shows that the network learned meaningful representations.

> The model treats characters with nearby embeddings as similar or partly interchangeable.

---

## 17. Increasing Embedding Size

A 2-D embedding may be a bottleneck. We are forcing 27 characters into only two coordinates.

Increase embedding dimension:

```python
C = torch.randn((27, 10))
```

Now each context of 3 characters produces:

```text
3 x 10 = 30 input values
```

So update the hidden layer:

```python
W1 = torch.randn((30, 200))
b1 = torch.randn(200)
W2 = torch.randn((200, 27))
b2 = torch.randn(27)
```

And flatten with:

```python
emb.view(-1, 30)
```

This model has around **11,000 parameters**.

After longer training and learning rate decay, Karpathy gets roughly:

```text
train loss: about 2.16
dev loss:   about 2.17 to 2.19
```

This confirms that the 2-D embedding was limiting performance.

### Hyperparameters to Tune

Karpathy invites the viewer to beat the best dev loss he got in about 30 minutes, around:

```text
2.17
```

Knobs to tune:

- `block_size`: number of previous characters used as context.
- `n_embd`: embedding dimensionality.
- `n_hidden`: hidden layer size.
- batch size.
- number of training steps.
- learning rate.
- learning rate decay schedule.
- initialization scale.
- regularization.

---

## 18. Sampling From the MLP Model

Sampling uses the same autoregressive idea as the bigram model, but now probability comes from the neural network.

### Sampling Algorithm

1. Start with context of all dots:

```python
context = [0] * block_size
```

2. Embed the context:

```python
emb = C[torch.tensor([context])]
```

The batch dimension is 1 because we generate one name at a time.

3. Forward through the network:

```python
h = torch.tanh(emb.view(1, -1) @ W1 + b1)
logits = h @ W2 + b2
```

4. Convert logits to probabilities:

```python
probs = F.softmax(logits, dim=1)
```

`F.softmax` does exponentiate-and-normalize, but safely, avoiding overflow.

5. Sample next character:

```python
ix = torch.multinomial(probs, num_samples=1, generator=g).item()
```

6. Shift context and append:

```python
context = context[1:] + [ix]
```

7. Stop when `ix == 0`.

### Full Sampler

```python
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
```

The generated names are much more name-like than the bigram model, though still imperfect.

Examples mentioned include outputs that start to sound like names, such as:

```text
ham...
joes...
```

The model is clearly better, but there is still much room to improve.

---

## 19. Final Code Skeleton

Below is a compact version of the final model pipeline.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3

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

g = torch.Generator().manual_seed(2147483647)
n_embd = 10
n_hidden = 200

C = torch.randn((27, n_embd), generator=g)
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for i in range(200000):
    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
```

Evaluation:

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'dev': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]
    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('dev')
```

---

## 20. Final Cheat Sheet

### Architecture

```text
context indices
    -> embedding lookup C
    -> flatten embeddings
    -> linear layer W1 + b1
    -> tanh
    -> linear layer W2 + b2
    -> logits
    -> cross entropy loss
```

### Shapes

For:

```python
block_size = 3
n_embd = 10
n_hidden = 200
batch_size = 32
```

| Tensor | Shape | Meaning |
|---|---:|---|
| `X[ix]` | `(32, 3)` | minibatch of contexts |
| `C` | `(27, 10)` | embedding table |
| `emb = C[X[ix]]` | `(32, 3, 10)` | embedded contexts |
| `emb.view(-1, 30)` | `(32, 30)` | flattened input |
| `W1` | `(30, 200)` | hidden-layer weights |
| `b1` | `(200,)` | hidden-layer bias |
| `h` | `(32, 200)` | hidden activations |
| `W2` | `(200, 27)` | output-layer weights |
| `b2` | `(27,)` | output-layer bias |
| `logits` | `(32, 27)` | next-character scores |
| `loss` | scalar | average NLL / cross entropy |

### Important PyTorch Lessons

- `C[X]` embeds every integer in `X`.
- `one_hot(i) @ C` is equivalent to `C[i]`, but indexing is faster.
- `.view(...)` reshapes by changing tensor metadata when possible; it is much cheaper than concatenation.
- `torch.cat(...)` creates a new tensor and copies memory.
- Broadcasting should always be checked by inspecting shapes.
- `F.cross_entropy(logits, targets)` is preferred over manual softmax + log + mean.
- Set `p.grad = None` before every backward pass.
- Use minibatches for speed; the gradient is noisy but useful.
- Use train/dev/test splits to avoid fooling yourself with training loss.

### Learning-Rate Workflow

1. Sweep learning rates exponentially:

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre
```

2. Plot loss vs learning-rate exponent.
3. Pick a value in the stable valley.
4. Train with that value.
5. Decay by 10x near the end.

### Model Improvement Knobs

- Increase `block_size`.
- Increase `n_embd`.
- Increase or decrease `n_hidden`.
- Tune batch size.
- Tune learning rate.
- Tune learning rate decay.
- Train longer.
- Improve initialization.
- Add regularization.

### Three Sentences That Summarise the Lecture

1. A fixed-context MLP language model learns embeddings for characters, concatenates the embeddings of previous characters, and uses a neural network to predict logits for the next character.
2. The model is trained with the same machinery as before: forward pass, `F.cross_entropy`, backpropagation, minibatch gradient descent, and learning-rate scheduling.
3. Compared with the bigram table, this framework scales: we can increase context size and network capacity without explicitly storing every possible context in a giant count table.

---

> **End of notes for Lecture 3 - Building Makemore, Part 2: MLP Character-Level Language Model.**
>
> The only part that changes as we scale toward modern models is the function that maps context to logits. The training objective and optimization loop stay fundamentally the same.
