# Neural Networks: Zero to Hero

## Lecture 2 — The Spelled-Out Intro to Language Modeling: Building Makemore (Bigram Edition)

> **Instructor:** Andrej Karpathy
> **Project:** [makemore](https://github.com/karpathy/makemore) — a character-level language-model playground.
> **Goal of this lecture:** Build a **bigram** character-level language model — *twice* — first by **counting**, then by **gradient-based learning on a tiny neural network**. Discover that the two approaches arrive at the **identical model**, then learn the framework that we will scale all the way up to GPT.

---

## Table of Contents

1. [What Is Makemore?](#1-what-is-makemore)
2. [Reading the `names.txt` Dataset](#2-reading-the-namestxt-dataset)
3. [What's a Bigram, and Why?](#3-whats-a-bigram-and-why)
4. [Counting Bigrams With a Python Dictionary](#4-counting-bigrams-with-a-python-dictionary)
5. [Counting Bigrams in a 2-D PyTorch Tensor](#5-counting-bigrams-in-a-2d-pytorch-tensor)
6. [Visualising the Bigram Matrix](#6-visualising-the-bigram-matrix)
7. [The `<S>`/`<E>` Token Trick — Use a Single `.`](#7-the-s-e-token-trick)
8. [Sampling Names From the Model](#8-sampling-names-from-the-model)
9. [Vectorised Normalisation & Broadcasting Pitfalls](#9-vectorised-normalisation--broadcasting-pitfalls)
10. [Evaluating the Model — Likelihood, Log-Likelihood, NLL](#10-evaluating-the-model--likelihood)
11. [Model Smoothing (Fake Counts)](#11-model-smoothing-fake-counts)
12. [Part 2 — The Neural-Network Approach](#12-part-2--the-neural-network-approach)
13. [Building the (x, y) Bigram Dataset for the Net](#13-building-the-bigram-dataset)
14. [One-Hot Encoding Integers](#14-one-hot-encoding-integers)
15. [A Single Linear Layer of Neurons via Matrix Multiplication](#15-a-single-linear-layer)
16. [Logits → Counts → Probabilities (Softmax)](#16-softmax)
17. [Vectorised Negative-Log-Likelihood Loss](#17-vectorised-loss)
18. [Backward Pass and Gradient-Descent Update](#18-backward-pass-and-update)
19. [Training Over the Whole Dataset](#19-training-over-the-whole-dataset)
20. [Insight #1 — One-Hot · W Just Plucks a Row of W](#20-insight-1--one-hot--w-just-plucks-a-row)
21. [Insight #2 — Smoothing = L2 Regularisation](#21-insight-2--smoothing--l2-regularisation)
22. [Sampling From the Trained Neural Net](#22-sampling-from-the-trained-neural-net)
23. [Final Cheat Sheet](#23-final-cheat-sheet)

---

## 1. What Is Makemore?

**makemore** is a tiny repo on Karpathy's GitHub that *makes more of things you give it*. Feed it a list of names (`names.txt` — about 32,000 names scraped from a government website) and after training it generates **new, unique, name-like strings** — perfect, for example, if you're trying to find a baby name.

Sample outputs after training the bigram model in this lecture:

```
dontel
irot
zhendi
…
```

Sound name-like, but they are *not* actual names from the dataset.

### A "Character-Level Language Model"

- Every example is a **line** (one word/name).
- Every line is treated as a **sequence of characters**.
- The model's job: **predict the next character** given the characters seen so far.

Throughout the playlist Karpathy will progressively replace the model's "brain":

> **bigram counts → bag-of-words → MLP → RNN/LSTM → Transformer (≡ GPT-2)**

The wrapping training/loss machinery never changes. **Only the function that maps `previous chars → logits` complexifies.**

---

## 2. Reading the `names.txt` Dataset

```python
words = open('names.txt', 'r').read().splitlines()

words[:10]      # ['emma', 'olivia', 'ava', 'isabella', 'sophia', ...]
len(words)      # 32033
min(len(w) for w in words)   # 2
max(len(w) for w in words)   # 15
```

So we have 32 k strings of lowercase letters, lengths 2–15.

> **Important framing.** A single word like `isabella` already contains *several* training examples for a character-level model:
>
> - `i` is likely to be the **first** character.
> - `s` is likely to come after `i`.
> - `a` is likely to come after `is`.
> - … and `isabella` is likely to **end**.
>
> Across 32 k names there is *a lot* of statistical structure to learn.

---

## 3. What's a Bigram, and Why?

A **bigram model** looks at exactly **one previous character** and predicts the next. It throws away all longer context. It is the **simplest, weakest** language model — and therefore a perfect place to start.

Iterate over consecutive character pairs of a word — a Python idiom worth remembering:

```python
for ch1, ch2 in zip(w, w[1:]):
    print(ch1, ch2)
```

`zip` halts at the shorter iterator, so `zip("emma", "mma")` produces `(e,m), (m,m), (m,a)`. Three bigrams from a 4-char word.

### We Need Special Start/End Tokens

`emma` only gives us inner bigrams. We also need to know:

- "**Words can start with `e`**"
- "**`a` is a likely *last* character**"

So we wrap the word in a special token `.` (originally Karpathy uses two: `<S>` and `<E>`, but later collapses them into one — see §7):

```python
chs = ['.'] + list(w) + ['.']
for ch1, ch2 in zip(chs, chs[1:]):
    print(ch1, ch2)
# . e
# e m
# m m
# m a
# a .
```

Now `.e` tells the model "e starts words" and `a.` says "a ends words".

---

## 4. Counting Bigrams With a Python Dictionary

The simplest way to "train" a bigram model is **just count**:

```python
b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# top bigrams by frequency:
sorted(b.items(), key=lambda kv: -kv[1])[:5]
# [(('n','.'), 6763), (('a','.'), 6640), (('a','n'), 5438), ...]
```

`n` is *very* often a **last** letter; `an` is one of the most common interior bigrams. So far so good — but a Python dictionary is awkward to do maths on. We need a 2-D **array**.

---

## 5. Counting Bigrams in a 2-D PyTorch Tensor

Switch to PyTorch:

```python
import torch
N = torch.zeros((27, 27), dtype=torch.int32)
```

Why **27 × 27**? 26 letters of the alphabet + 1 special `.` token. **Rows = first character of bigram, columns = second character.** Cell `(i, j)` will hold the count of bigram `(char_i, char_j)`.

### Mapping Characters ↔ Integers

Tensors only accept integer indices, so we build lookup tables.

```python
chars = sorted(list(set(''.join(words))))   # ['a', 'b', ..., 'z']
stoi = {s: i+1 for i, s in enumerate(chars)}   # 'a':1, 'b':2, ..., 'z':26
stoi['.'] = 0                                  # '.' goes to index 0
itos = {i: s for s, i in stoi.items()}         # reverse map
```

Putting `.` at **index 0** is purely cosmetic — Karpathy likes `a` to start at 1, the rows then read `., a, b, c, …` which is more natural.

### Filling the Counts Matrix

```python
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```

Now `N[i, j]` is the count of bigram (`itos[i]`, `itos[j]`).

---

## 6. Visualising the Bigram Matrix

A raw 27×27 grid of numbers is hard to read. Karpathy writes a `matplotlib` snippet that paints each cell with its bigram label *and* count:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
```

> ⚠️ `N[i, j]` returns a **0-dim tensor**, not a Python integer. Call `.item()` to extract the scalar.

The picture clearly shows structure: the `.` row (word starts) is dominated by common first letters (`a, m, j, k, …`); the `.` column (word ends) is dominated by `a, n, e, …`; and the body shows letter-pair frequencies.

---

## 7. The `<S>`/`<E>` Token Trick

Karpathy's first attempt used **two** special tokens, `<S>` and `<E>`, giving a 28×28 matrix. But two slots is wasteful:

- The `<E>` row is **all zeros** (an end-token can never be the *first* character of a bigram).
- The `<S>` column is **all zeros** (a start-token can never be the *second* character).

So we collapse them into **one** token (`.`) used for *both* start and end. The matrix shrinks to **27×27** with no loss of information.

> 🎯 **Worth absorbing.** In a real-world tokenizer designer's life, every cell of a vocab matters. A model that uses *two* special tokens here would just waste capacity.

---

## 8. Sampling Names From the Model

The model is *just the matrix* `N`. To **sample** a name:

1. Start at row `0` (the `.`/start row).
2. Convert the row to a probability distribution: `p = N[ix].float(); p = p / p.sum()`.
3. Draw a random index from `p` with `torch.multinomial`.
4. Print the corresponding character.
5. Move to that row and repeat — until we sample `0` (the end token), then break.

### `torch.multinomial`

> "Give me probabilities; I'll give you integers sampled according to those probabilities."

```python
g = torch.Generator().manual_seed(2147483647)   # deterministic RNG
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
```

### Full Sampler

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))
```

Sample outputs:

```
mor
axx
minaymoryles
kondlaisah
anchshizarie
…
```

> "I'll be honest with you, this doesn't look right. … But it actually is. The bigram model is just *terrible*." — Karpathy

For comparison, sampling from a *uniform* model (`p = torch.ones(27)/27`) gives unintelligible garbage. So our model has *clearly* learned something — it's just that one character of context is not enough to produce English-like names.

---

## 9. Vectorised Normalisation & Broadcasting Pitfalls

Re-normalising one row at every sampling step is wasteful. Compute the **whole** probability matrix `P` once, up front, then sample by `P[ix]`.

```python
P = N.float()
P /= P.sum(dim=1, keepdim=True)
```

### Why `dim=1, keepdim=True`?

`N` has shape `(27, 27)`. `N.sum(dim=1)` sums along the **second** axis — i.e. **summing each row** — giving a 27-vector of row totals.

- With `keepdim=True` → shape `(27, 1)` (a column vector).
- With `keepdim=False` → shape `(27,)` (squeezed; no row dimension left).

Now the division `(27,27) / (27,1)` is allowed by **broadcasting** because, scanning right-to-left, dimensions are either equal *or* one of them is `1`. PyTorch *replicates* the column vector across all 27 columns and does element-wise division.

### The Subtle Bug

If we **forget** `keepdim=True`, the divisor has shape `(27,)`. Broadcasting will *prepend* a `1`, making it `(1, 27)` — a **row** vector. Then `(27, 27) / (1, 27)` replicates the row vector *vertically* and divides — meaning we **normalise the columns instead of the rows!**

> ⚠️ *The numbers happen to look the same in our case because the bigram count matrix is roughly symmetric for unrelated reasons.* Karpathy stresses this is one of the most insidious classes of bug:
>
> 1. The code runs.
> 2. The numbers look plausible.
> 3. Yet you've silently done the wrong operation.
>
> **Always check `keepdim`. Always read the broadcasting rules. Always inspect shapes.**

Final, in-place version:

```python
P = (N + 1).float()                 # (+1 = smoothing, see §11)
P /= P.sum(dim=1, keepdim=True)     # in-place — no new tensor
```

---

## 10. Evaluating the Model — Likelihood

We have a model. Is it **good**? We need *one* number that summarises quality.

For each bigram `(ch1, ch2)` in the dataset, the model assigns a probability `P[ix1, ix2]`. The classical statistical objective is **likelihood**:

\[ \text{likelihood} = \prod_i P_i \]

The model with the *highest* likelihood on the training set is the best fit. But:

- Probabilities are in `(0, 1)`, so the product of thousands of them **underflows to ~0** in floating point.
- Multiplications are awkward.

So we work with **log-likelihood** instead. Logarithm turns products into sums:

\[ \log \prod_i P_i = \sum_i \log P_i \]

`log(p)` is monotonically increasing, peaks at `0` (when `p=1`), and goes to `−∞` as `p → 0`.

### From Likelihood to a *Loss*

A loss should be *low when good*. Two transformations:

1. **Negate** → "negative log-likelihood" (NLL). Now `0` is best, and worse models give larger positive numbers.
2. **Average** over examples (so the number doesn't depend on dataset size).

```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood
print(f"loss = {nll/n:.4f}")     # ≈ 2.4541
```

### Read-out

- A **uniform** model would give `log(1/27) ≈ −3.30` per bigram → loss ≈ 3.30.
- Our **counted bigram** model gets ≈ **2.45** — clearly better than chance, but still bad.
- A *perfect* model would give NLL = 0.

> 📌 **Remember.** The "training loss" of a language model and its "average per-character negative log-likelihood" are the **same thing**.

---

## 11. Model Smoothing (Fake Counts)

Try evaluating an unusual name like `andrejq`:

```python
# bigram 'jq' has count 0  ⇒  prob = 0  ⇒  log(0) = −∞  ⇒  loss = ∞
```

A single zero-count bigram makes the loss **infinite**. That's a brittle model.

### The Fix — Add a Constant to All Counts

```python
P = (N + 1).float()
P /= P.sum(dim=1, keepdim=True)
```

Adding `1` (or any positive constant `α`) to every count guarantees all probabilities are positive.

- `α → 0`: peaked, brittle distribution → matches the data exactly but goes infinite on unseen bigrams.
- `α → ∞`: every row becomes uniform → throws away all signal.
- A small `α` in between is the sensible middle ground.

This is a 200-year-old trick called **Laplace / additive smoothing**. It will reappear later in §21 disguised as **L2 regularisation** in the neural-network framework.

---

## 12. Part 2 — The Neural-Network Approach

We have a working bigram model. Now redo it as a neural net.

> **Why bother?** Because the *counting* approach **does not scale**. With 1 previous char we need a 27×27 table (729 cells). With 5 previous chars we'd need a 27⁵-row table (14 million cells), most of them empty. With 10 previous chars (still ridiculously short for a real language model) the table doesn't fit in any computer.
>
> **Neural networks compress** all that information into a small set of weights and *generalise* across contexts they've never seen.

### The Plan

```
input character (integer)
  → one-hot vector (27 dims)
  → linear layer  ( × W:  27 → 27 )
  → logits  (27 numbers)
  → exponentiate → counts → normalise → probabilities
  → negative-log-likelihood loss against the true next character
  → backward pass → gradient on W → SGD update
```

It will exactly mirror micrograd's training loop.

---

## 13. Building the Bigram Dataset for the Net

Instead of *counting* bigrams, *list* them as `(input_int, target_int)` pairs:

```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples:', num)    # 228,146 for full dataset
```

For just `emma` we get 5 examples: `(0→5)`, `(5→13)`, `(13→13)`, `(13→1)`, `(1→0)`.

> ⚠️ **`torch.tensor` vs `torch.Tensor`** (lowercase vs capital).
> - `torch.tensor(xs)` infers `dtype`. Integers stay as `int64`. *Use this.*
> - `torch.Tensor(xs)` always returns `float32`.
> - The docs are unclear; the difference is buried in a forum thread.
> - **Habit:** always use lowercase `torch.tensor`.

---

## 14. One-Hot Encoding Integers

We can't feed integer indices straight into a neural net — they would mix multiplicatively with the weights and the value of "13" wouldn't mean anything semantically larger than "5".

Instead, represent each integer `i` as a length-27 vector with a `1` in position `i` and `0`s everywhere else:

```python
import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27).float()
# xenc.shape == (num_examples, 27)
```

> ⚠️ `F.one_hot` returns `int64`. Neural nets need **floats**, so cast to `.float()`. (Surprisingly, `one_hot` doesn't accept a `dtype` argument.)

Visualising one row of `xenc` shows a single bright cell — exactly the integer encoded.

---

## 15. A Single Linear Layer of Neurons via Matrix Multiplication

A layer with `27` input dims and `27` output neurons is just a `(27, 27)` weight matrix:

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

`torch.randn` draws from a standard normal (mean 0, std 1).

The forward pass for *all* examples *and all* neurons in one call:

```python
logits = xenc @ W       # (N, 27) @ (27, 27)  =  (N, 27)
```

### What Just Happened?

- `(num_examples, 27) @ (27, 27)` → `(num_examples, 27)`.
- Element `[i, j]` of the result = (i-th input one-hot row) · (j-th column of W) = the firing rate of **neuron j** on **example i**.
- All neurons evaluated on all examples **in parallel**. This is *the* reason GPUs and tensors exist.

You can convince yourself of the equivalence:

```python
# the [3, 13] entry of logits is identical to:
(xenc[3] * W[:, 13]).sum()
```

> No bias, no non-linearity yet — just a plain linear layer. We'll explain *why* this is enough in §16.

---

## 16. Softmax — Logits → Counts → Probabilities

The 27 outputs are real numbers (positive *and* negative). Probabilities must be **non-negative** and **sum to 1**. So we interpret the network's outputs as **log-counts** ("logits") and run them through:

```python
counts = logits.exp()                      # exp(x) is always positive
probs  = counts / counts.sum(1, keepdim=True)   # rows sum to 1
```

The two-step `exp` then `normalise` operation is universally called **softmax**:

\[ \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \]

Properties of softmax:

- Output is always a valid probability distribution (positive, sums to 1).
- Larger logits → larger probabilities, but smoothly differentiable.
- Used as the **final layer of every classifier** you'll ever meet.

### The Whole Forward Pass

```python
logits = xenc @ W                           # (N, 27)
counts = logits.exp()                       # (N, 27)
probs  = counts / counts.sum(1, keepdim=True)  # (N, 27)
```

Every step is **differentiable**. Therefore PyTorch (just like micrograd!) tracks the entire computation graph and we can call `.backward()` on a final scalar.

---

## 17. Vectorised Negative-Log-Likelihood Loss

For each example `i`, we want the probability the network assigned to the **correct** next character `ys[i]`. Indexing trick:

```python
loss = -probs[torch.arange(num), ys].log().mean()
```

Reading this from the inside out:

- `torch.arange(num)` → `[0, 1, 2, …, num-1]` (one row index per example).
- `probs[torch.arange(num), ys]` → length-`num` vector. Row `i` returns `probs[i, ys[i]]` — the probability assigned to the true label.
- `.log()` → take the log of each.
- `.mean()` → average across examples.
- `-` → flip sign so that lower is better.

This single line is **the entire negative-log-likelihood loss for a classifier**. You will see it again in every neural-net training script for the rest of your life.

---

## 18. Backward Pass and Gradient-Descent Update

Identical pattern to micrograd's MLP training loop:

```python
# 1. forward pass
xenc   = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs  = counts / counts.sum(1, keepdim=True)
loss   = -probs[torch.arange(num), ys].log().mean()

# 2. backward pass
W.grad = None                # PyTorch idiom; equivalent to zeroing
loss.backward()              # autograd fills W.grad

# 3. update
W.data += -50 * W.grad       # learning rate 50 (large because the problem is easy)
```

A few subtleties:

- `requires_grad=True` was set when we created `W`. By default leaf tensors do **not** track gradients (efficiency).
- `W.grad = None` is the PyTorch-recommended way to "zero the gradient". `None` is faster than allocating a zero tensor and is interpreted by the optimiser as "this gradient hasn't been written yet".
- The **first** forward pass with random `W` gives loss ≈ **3.76**. The first update lowers it slightly. Repeat.

---

## 19. Training Over the Whole Dataset

Plug it into a loop and run for a hundred steps with `lr = 50`:

```python
for k in range(100):
    # forward
    xenc   = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs  = counts / counts.sum(1, keepdim=True)
    loss   = -probs[torch.arange(num), ys].log().mean()

    # backward
    W.grad = None
    loss.backward()

    # update
    W.data += -50 * W.grad
    print(loss.item())
```

After ~100 steps loss converges to **≈ 2.46**, *exactly* the same number we got from pure counting + Laplace smoothing.

> 🤯 **The two completely different methods produce the same loss because they produce the same model.** See Insight #1.

---

## 20. Insight #1 — One-Hot · W Just Plucks a Row of W

When the input is a one-hot vector with a `1` at position `k`, the matrix product `xenc @ W` simply selects **row `k` of W**.

So:

- The **counting** approach stored bigram statistics directly in a `(27, 27)` table `P`, where row `i` was the probability distribution over the next character given that the previous character was `i`.
- The **neural** approach stores `W` as a `(27, 27)` matrix; the relevant row is plucked out by the one-hot, then exponentiated and normalised.

In other words, **`exp(W)` after normalisation is the equivalent of `P`**. The two models have *the same shape and the same expressive power.* The only difference is *how we got there*: counting vs. gradient descent.

Counting is closed-form and instant. Gradient descent is iterative and slower — but it generalises to far more complex networks where no closed-form solution exists.

---

## 21. Insight #2 — Smoothing = L2 Regularisation

We added `+1` counts in §11 to avoid `log(0)`. There's an exact analogue in the neural framework: **L2 regularisation**.

If `W = 0`, then `logits = xenc @ W = 0`, so `counts = 1`, so `probs` are uniform. **A zero `W` produces a perfectly uniform model — exactly what infinite smoothing produces.**

So adding a "spring force pulling `W` toward `0`" inside the loss function is *equivalent* to adding fake counts:

```python
loss = -probs[torch.arange(num), ys].log().mean() \
       + 0.01 * (W**2).mean()
```

- `(W**2).mean()` is `0` only when `W` is all zeros, and grows otherwise.
- The coefficient `0.01` controls regularisation strength — analogous to the constant `α` we add to counts.
- Stronger regularisation → smoother (more uniform) predictions.

> 🎯 **Same idea, two languages.** Statisticians call it *smoothing*. Deep-learning practitioners call it *L2 / weight-decay*. Mathematically identical.

---

## 22. Sampling From the Trained Neural Net

Sampling looks **almost identical** to §8, except the row of probabilities now comes from the network rather than from `P`:

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        # forward pass on the single character ix
        xenc   = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p      = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1,
                               replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))
```

Outputs are **bit-for-bit identical** to the counting model with the same RNG seed — confirming once again that we trained the same model two different ways.

---

## 23. Final Cheat Sheet

### The Bigram-Modelling Pipeline (Counting)

| Step | Code |
|------|------|
| Read words | `words = open('names.txt').read().splitlines()` |
| Vocab maps | `stoi`, `itos` (with `.` at index 0) |
| Build counts | `N[ix1, ix2] += 1` for every bigram |
| Smooth | `N + 1` |
| Normalise rows | `P = N.float(); P /= P.sum(1, keepdim=True)` |
| Sample | `torch.multinomial(P[ix], 1, replacement=True, generator=g)` |
| Evaluate (loss) | `-mean(log(P[ix1, ix2]))` over all bigrams |

### The Bigram-Modelling Pipeline (Neural Net)

| Step | Code |
|------|------|
| Encode input | `xenc = F.one_hot(xs, 27).float()` |
| Forward | `logits = xenc @ W; counts = logits.exp(); probs = counts / counts.sum(1, keepdim=True)` |
| Loss | `-probs[torch.arange(num), ys].log().mean() + α * (W**2).mean()` |
| Backward | `W.grad = None; loss.backward()` |
| Update | `W.data += -lr * W.grad` |
| Sample | as above, but `p` comes from running one input through the net |

### Concept Map (Counting ↔ NN)

| Counting world | Neural world |
|----------------|--------------|
| `P` (27 × 27 prob matrix) | `softmax(W)` row-wise |
| Bigram count `N[i,j]` | Logit `W[i,j]` (≈ `log` of count) |
| `+1` smoothing | `+α‖W‖²` regularisation |
| Closed-form normalisation | Iterative gradient descent on NLL |

### Common Bugs Highlighted in This Lecture

- ❌ **Forgetting `keepdim=True`** when summing — broadcasts the wrong way and silently normalises columns instead of rows.
- ❌ **Forgetting `.float()`** when one-hot-encoding — feeds integers to a layer that needs floats.
- ❌ **Confusing `torch.tensor` vs `torch.Tensor`** — first preserves dtype, second always casts to float.
- ❌ **Forgetting `.item()`** when extracting a scalar from a 0-dim tensor.
- ❌ **Forgetting `requires_grad=True`** on weights — `loss.backward()` will produce no gradient.
- ❌ **Forgetting to zero gradients** before each backward pass (same lesson as micrograd Lecture 1).

### Three Sentences That Summarise the Whole Lecture

1. A bigram language model is just a `(27, 27)` table of probabilities — and you can build it either by counting or by gradient-descending against the negative-log-likelihood of the data.
2. The neural-net framework — one-hot encode → linear layer → softmax → NLL loss → backward → update — produces the exact same model as counting, only via a different path.
3. The *real* prize is that the neural-net framework **scales**: as we feed in more previous characters and replace the linear layer with deeper networks (MLP → RNN → Transformer), the surrounding training machinery does not change one bit.

---

> **End of notes for Lecture 2 — *Building Makemore: The Bigram Edition*.**
>
> *"None of this will fundamentally change. The only thing that will change is the way we do the forward pass. We'll use the same machinery to optimise it — all the way up to Transformers."* — Andrej Karpathy
