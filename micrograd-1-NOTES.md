# Neural Networks: Zero to Hero

## Lecture 1 — The Spelled-Out Intro to Neural Networks and Backpropagation: Building Micrograd

> **Instructor:** Andrej Karpathy
> **Project:** [micrograd](https://github.com/karpathy/micrograd) — a tiny scalar-valued autograd engine + a small neural net library on top of it.

---

## Table of Contents

1. [What Is Micrograd?](#1-what-is-micrograd)
2. [Derivatives — The Real Intuition](#2-derivatives--the-real-intuition)
3. [Functions of Multiple Inputs](#3-functions-of-multiple-inputs)
4. [The `Value` Object — Wrapping a Scalar](#4-the-value-object--wrapping-a-scalar)
5. [Tracking the Expression Graph (`_prev`, `_op`)](#5-tracking-the-expression-graph)
6. [Visualizing the Graph (Graphviz)](#6-visualizing-the-graph)
7. [Manual Backpropagation #1 — A Simple Expression](#7-manual-backpropagation-1--a-simple-expression)
8. [The Chain Rule](#8-the-chain-rule)
9. [A Single Optimization Step (Preview)](#9-a-single-optimization-step-preview)
10. [Manual Backpropagation #2 — A Single Neuron with `tanh`](#10-manual-backpropagation-2--a-single-neuron-with-tanh)
11. [Automating Backward — Per-Operation `_backward`](#11-automating-backward--per-operation-_backward)
12. [Whole-Graph Backward — Topological Sort](#12-whole-graph-backward--topological-sort)
13. [The Multi-Use Variable Bug — Accumulating Gradients](#13-the-multi-use-variable-bug)
14. [Breaking `tanh` Into Atoms — `exp`, `pow`, `div`, `sub`](#14-breaking-tanh-into-atoms)
15. [Same Thing in PyTorch](#15-same-thing-in-pytorch)
16. [Building a Neural Net Library — `Neuron`, `Layer`, `MLP`](#16-building-a-neural-net-library)
17. [A Tiny Dataset and the Loss Function](#17-a-tiny-dataset-and-the-loss-function)
18. [Gradient Descent — Training the Network](#18-gradient-descent--training-the-network)
19. [The `zero_grad` Bug — A Classic Mistake](#19-the-zero_grad-bug)
20. [Summary — From Micrograd to Modern Neural Nets](#20-summary--from-micrograd-to-modern-neural-nets)
21. [Walkthrough of the Real Micrograd Repo](#21-walkthrough-of-the-real-micrograd-repo)
22. [Glimpse Into Production — PyTorch's `tanh` Backward](#22-glimpse-into-production--pytorchs-tanh-backward)
23. [Final Cheat Sheet](#23-final-cheat-sheet)

---

## 1. What Is Micrograd?

**Micrograd** is a tiny autograd engine.

- **autograd** = "automatic gradient" → it implements **backpropagation**.
- **Backpropagation** = an algorithm that efficiently evaluates the **gradient of a loss function with respect to the weights of a neural network**.
- Knowing those gradients lets us **iteratively tune the weights** to *minimise the loss*, which improves the network's accuracy.
- Backprop is the **mathematical core** of every modern deep-learning library — **PyTorch, JAX, TensorFlow** — they are all just very fast, GPU-aware implementations of the same idea.

### Two Big Insights

1. **A neural network is just a (possibly huge) mathematical expression.** It takes input data and weights, runs some maths, and outputs a prediction (or a loss).
2. **Backpropagation is more general than neural nets.** It works on *any* mathematical expression. We just happen to use it for training networks.

### What Makes Micrograd Special

- **~100 lines for the engine** (`engine.py`) and **~50 lines for the neural-net library** (`nn.py`) — about **150 lines total** for everything you need to train a neural network.
- Everything else in real-world libraries (PyTorch, JAX) is just **efficiency** — using **n-dimensional tensors** instead of scalars so we can parallelise on CPU/GPU. *None of the maths changes.*
- Micrograd is a **scalar-valued** engine — it operates on individual numbers (e.g. `-4.0`, `2.0`). It is intentionally inefficient for **pedagogical** reasons: you literally see every `+` and every `*`, every chain-rule application.

### The Promise of the Lecture

> "Micrograd is what you need to train neural networks. Everything else is just efficiency." — Karpathy

By the end you will:

- Build a `Value` class that wraps a scalar and remembers how it was produced.
- Implement **forward pass**, **backward pass**, and the **chain rule** by hand.
- Build a **multi-layer perceptron (MLP)** out of those values.
- Train it with **gradient descent** on a toy dataset.
- Understand exactly what happens inside PyTorch when you call `loss.backward()`.

---

## 2. Derivatives — The Real Intuition

Before writing any code, we need a *gut feeling* for what a derivative *is*. Forget symbolic rules for a moment.

### A Simple Function

```python
def f(x):
    return 3*x**2 - 4*x + 5

f(3.0)   # → 20.0
```

It's a parabola. Its shape can be plotted by feeding `np.arange(-5, 5, 0.25)` into it and using `matplotlib`.

### The Definition (Numerically)

In a calculus class you'd derive `f'(x) = 6x - 4` symbolically. **In neural nets nobody ever writes the symbolic derivative** — the expression has thousands or millions of terms. We approach it **numerically** instead.

The textbook definition is:

\[ f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \]

In words: **bump `x` by a tiny amount `h`; how does the function respond?** That sensitivity *is* the derivative — the slope at that point.

```python
h = 0.000001
x = 3.0
slope = (f(x + h) - f(x)) / h
# ≈ 14.0   (matches 6·3 − 4 = 14)
```

### Sign Tells You The Direction

- At `x = 3` → slope is `+14`. Bumping `x` up makes `f` grow.
- At `x = −3` → slope is **negative** (the function decreases when you go right).
- At `x = 2/3` → slope is **0**. That's the bottom of the parabola — nudging `x` doesn't change `f` (locally).

> ⚠️ **Floating-point caution.** If `h` is *too* small (e.g. `1e-15`), you stop converging because of finite-precision arithmetic. `1e-3` to `1e-6` works fine.

---

## 3. Functions of Multiple Inputs

Now the function has **three inputs** `a`, `b`, `c` and one output `d`:

```python
a = 2.0
b = -3.0
c = 10.0
d = a*b + c           # → 4.0
```

To find ∂d/∂a numerically:

```python
h = 0.0001
a2 = a + h
d2 = a2*b + c
slope = (d2 - d) / h    # → -3.0  (this is just b)
```

### Intuition Per Variable

| Input | Bump direction | What happens to `d` | Slope |
|-------|----------------|---------------------|-------|
| `a` | `+h` | `b` is negative, so `a*b` shrinks → `d` shrinks | **`b = −3`** |
| `b` | `+h` | `a` is positive, so `a*b` grows → `d` grows | **`a = +2`** |
| `c` | `+h` | `c` is added directly | **`+1`** |

These match the symbolic answer:
- ∂(`a·b + c`)/∂`a` = `b`
- ∂(`a·b + c`)/∂`b` = `a`
- ∂(`a·b + c`)/∂`c` = `1`

The derivative tells you **how strongly that input influences the output, and in which direction**. This is *exactly* what we need to tune neural-net weights.

---

## 4. The `Value` Object — Wrapping a Scalar

Neural-net expressions are huge, so we need a **data structure** to hold them. Start with a class that wraps a single scalar:

```python
class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"
```

Why `__repr__`? So that `print(Value(2.0))` shows `Value(data=2.0)` instead of an ugly memory address.

### Adding Operators

Python lets you overload arithmetic via dunder methods:

```python
def __add__(self, other):
    out = Value(self.data + other.data)
    return out

def __mul__(self, other):
    out = Value(self.data * other.data)
    return out
```

So `a + b` becomes `a.__add__(b)`, returning a **new `Value`** wrapping the sum. After this we can do:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c       # Value(data=4.0)
```

---

## 5. Tracking the Expression Graph

A `Value` must remember **where it came from** — its parents (children in graph terminology) and the operation that produced it. We add two private fields:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)   # set of parent Value objects
        self._op = _op                # the operation that created this node

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

Now every `Value` knows:

- `data`: its scalar value
- `_prev`: the parents (a *set* — used for efficiency)
- `_op`: the operator that produced it (`'+'`, `'*'`, …)

So given `d = a*b + c`:

- `d._op == '+'`
- `d._prev == {Value(a*b), Value(c)}`

We've built a **directed acyclic graph (DAG)** — a *computation graph* — that records the entire forward pass.

---

## 6. Visualizing the Graph

Karpathy uses **Graphviz** (`pip install graphviz`) to draw the DAG. The helper `draw_dot(root)` does two things:

1. `trace(root)` walks the graph and returns the set of all nodes and edges.
2. It builds a Graphviz `Digraph`, drawing each `Value` as a rectangle and each operation as a small ellipse between a node and its parents.

We also added a `label` field so each variable shows its name (`a`, `b`, `e = a*b`, `d = e + c`, …).

Once we extend the example one layer deeper (`L = d * f`) we have:

```
a ─┐
   ├─(*) → e ─┐
b ─┘          ├─(+) → d ─┐
       c ─────┘          ├─(*) → L
                  f ─────┘
```

This is the **forward pass**. The output is `L = -8.0`.

---

## 7. Manual Backpropagation #1 — A Simple Expression

We add a `grad` attribute to every `Value` (default `0.0`). Semantically:

> `node.grad` = the partial derivative **of the final output `L`** with respect to *this node*.

Initialised to `0` because — by default — we assume the variable has *no effect* on the output.

We now fill in gradients **manually**, walking *backwards* from the output.

### Step 0 — Base Case

\[ \frac{\partial L}{\partial L} = 1 \]

So `L.grad = 1.0`.

### Step 1 — `L = d · f`

We learnt earlier that for a product `L = d·f`:
- ∂L/∂d = `f`
- ∂L/∂f = `d`

Hence:
- `d.grad = f.data = -2.0`
- `f.grad = d.data =  4.0`

Karpathy verifies this *numerically* by bumping `f.data += h` and recomputing `L` — the slope is indeed `4`.

### Step 2 — `d = e + c`

Inside this little `+` node, we only know:
- ∂d/∂e = 1
- ∂d/∂c = 1

But we want ∂L/∂e and ∂L/∂c — *not* ∂d/∂e. To bridge the gap we need the **chain rule**.

---

## 8. The Chain Rule

If `z` depends on `y`, and `y` depends on `x`, then:

\[ \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} \]

> **Intuition (Wikipedia's car/bicycle/man story).** If a car goes 2× as fast as a bicycle, and the bicycle goes 4× as fast as a walking man, then the car goes 2 × 4 = 8× as fast as the man. We just **multiply the rates**.

### Applying It Inside Our Graph

For the `+` node `d = e + c`:

\[ \frac{\partial L}{\partial c} = \underbrace{\frac{\partial L}{\partial d}}_{\text{global}} \cdot \underbrace{\frac{\partial d}{\partial c}}_{\text{local}} = (-2) \cdot 1 = -2 \]

Same for `e`. So:

> **A `+` node simply *routes* the incoming gradient unchanged to all its children.**

Now backprop into `e = a · b` (a `*` node). The local derivatives are:
- ∂e/∂a = `b = -3`
- ∂e/∂b = `a =  2`

Chain rule again:
- `a.grad = (∂L/∂e) · (∂e/∂a) = (-2)·(-3) = 6`
- `b.grad = (∂L/∂e) · (∂e/∂b) = (-2)·( 2) = -4`

Karpathy verifies all of these numerically. They match.

### The Big Picture

Backpropagation is **a recursive application of the chain rule**, walking backwards through the computation graph. At each node:

1. Start with `out.grad` (already filled in by the children).
2. Multiply by the **local derivative** of the operation.
3. **Add** the result into each parent's `.grad`.

That is the whole algorithm.

---

## 9. A Single Optimization Step (Preview)

If `a.grad = 6`, then increasing `a` by a small step **increases** `L`. To **maximise** `L` we move *along* the gradient; to **minimise** it we move *against* it.

```python
step = 0.01
a.data += step * a.grad
b.data += step * b.grad
c.data += step * c.grad
f.data += step * f.grad

# Now recompute the forward pass:
e = a * b
d = e + c
L = d * f
print(L.data)   # was -8.0, now ≈ -7.0  (went *up*)
```

This is the seed of **gradient descent** — we'll come back to it for real once we have a full neural net.

---

## 10. Manual Backpropagation #2 — A Single Neuron with `tanh`

### A Mathematical Model of a Neuron

Biological neurons are wildly complex. The simple computational model is:

\[ \text{out} = \varphi\!\Big(\sum_i w_i x_i + b\Big) \]

- `x_i` — inputs.
- `w_i` — synaptic weights (one per input).
- `b` — bias (the neuron's "trigger-happiness").
- `φ` — an **activation function**, typically a **squashing function** like `tanh`, `sigmoid` or `ReLU`.

`tanh` looks like a smooth S-curve: it squashes any real number into `(-1, 1)`. Plotted with `np.tanh`:

```
        +1 ─────────
              ╱
             ╱
   ─────── 0
           ╱
          ╱
   ─────── −1
```

### Wiring Up A Tiny Neuron in Micrograd

```python
x1 = Value(2.0,  label='x1')
x2 = Value(0.0,  label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0,  label='w2')
b  = Value(6.8813735870195432, label='b')   # chosen so numbers come out nicely

x1w1   = x1 * w1
x2w2   = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n      = x1w1x2w2 + b      # raw cell-body activation
o      = n.tanh()          # final output
```

(The bias `6.8813…` is chosen so that the final `o` ≈ `0.7071` — a clean number for arithmetic.)

### Implementing `tanh` On `Value`

`tanh` involves exponentiation, which we haven't implemented yet — but we don't have to break it down into atoms. We can treat it as a **single operation** as long as we know its derivative.

```python
import math

def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')
    return out
```

> **Key takeaway.** *The "atomic" level of operations is up to you.* You can backprop through `+`, `*`, *or* through a giant block called `tanh`. All you need is a way to compute the **local derivative** of that block.

### Manual Backprop Through The Neuron

The derivative of `tanh`:

\[ \frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) \]

So if `o = tanh(n)`, then ∂o/∂n = `1 - o.data²`. With `o.data ≈ 0.7071`, that's `1 - 0.5 = 0.5`. Conveniently round.

| Node | grad | Why |
|------|------|-----|
| `o` | **1.0** | base case ∂o/∂o |
| `n` | **0.5** | local `1 - o.data²` × outer `1.0` |
| `b`, `x1w1x2w2` | **0.5** each | `+` node distributes gradient |
| `x1w1`, `x2w2` | **0.5** each | another `+` node |
| `x2` | `w2.data · 0.5 = 0.5` | `*` node, swap-and-multiply |
| `w2` | `x2.data · 0.5 = 0.0` | `x2` was `0`, so this weight has **no effect** right now |
| `x1` | `w1.data · 0.5 = -1.5` | |
| `w1` | `x1.data · 0.5 = 1.0` | |

Notice how `x2 = 0` zeroes out the gradient on `w2`. That makes sense: changing a weight that multiplies a zero input cannot affect the output — so the partial derivative *must* be zero.

---

## 11. Automating Backward — Per-Operation `_backward`

Manual backprop is tedious. We let each `Value` know **how to push its gradient back** by storing a closure.

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None        # default: do nothing (leaves)
```

### `__add__`

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad  += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
```

A `+` node simply **routes** `out.grad` into both inputs (local derivative is `1`).

### `__mul__`

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad
    out._backward = _backward
    return out
```

A `*` node **swaps** the two operands and multiplies — that's the local derivative.

### `tanh`

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
```

Local derivative `1 − tanh²(x)` × the upstream gradient — chain rule again.

### Running It

```python
o.grad = 1.0
o._backward()
n._backward()
x1w1x2w2._backward()
b._backward()
x1w1._backward()
x2w2._backward()
```

We get the *same* gradients we computed by hand. ✅

> ⚠️ **Common bug.** Don't write `out._backward = _backward()` (with parentheses). You'd be calling the function and storing `None`. We want to **store the function itself**.

---

## 12. Whole-Graph Backward — Topological Sort

Calling `_backward` on every node by hand is still annoying. We need an automatic order. The rule is:

> **A node's `_backward` may run only after every node that depends on it has already run.**

This is exactly **topological sort** of the DAG. Build the order recursively:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)        # add AFTER children → invariant
    build_topo(self)

    self.grad = 1.0               # base case
    for node in reversed(topo):   # walk backwards
        node._backward()
```

Now we just call `loss.backward()` and **the whole graph propagates gradients automatically.**

> **Mental model.** The forward pass *grows* a graph from leaves to root. The backward pass *floods* gradient information from the root back down to the leaves, applying the local chain rule at every node.

---

## 13. The Multi-Use Variable Bug

Consider:

```python
a = Value(3.0)
b = a + a            # a is used twice
b.backward()
print(a.grad)        # → 1.0  ❌ should be 2.0
```

The issue: inside `__add__`, `self` and `other` are the *same object*, so we **overwrite** `a.grad` with `1.0` twice instead of adding `1.0 + 1.0`.

A subtler example: in `f = (a*b) * (a + b)`, the variable `a` flows through both `*` and `+`. When two branches of the graph send gradient back to the same leaf, **we must accumulate, not overwrite**.

### The Fix — Use `+=` Everywhere

In every `_backward`, replace `=` with `+=`:

```python
self.grad  += 1.0 * out.grad
other.grad += 1.0 * out.grad
```

This works because `grad` starts at `0`, so the first contribution lands correctly, and subsequent ones add on top.

> Mathematically this is the **multivariate chain rule**: when a variable enters the output through multiple paths, the total derivative is the **sum** of the path-derivatives.

---

## 14. Breaking `tanh` Into Atoms

`tanh` was a single op. Just to flex the engine, let's express it as:

\[ \tanh(n) = \frac{e^{2n} - 1}{e^{2n} + 1} \]

To rebuild it we need: **exponentiation**, **division**, **subtraction**, and a way to add a plain Python number to a `Value`.

### Allow `Value` ↔ `int/float` Mixing

In `__add__` and `__mul__` we wrap the other operand if it isn't already a `Value`:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    ...
```

But `2 * a` doesn't trigger `a.__mul__(2)` — Python tries `(2).__mul__(a)` first, which fails. We add a fallback:

```python
def __rmul__(self, other):   # called when 2*a fails on the int
    return self * other
```

Same trick for `__radd__`.

### `exp`

```python
def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
        self.grad += out.data * out.grad   # d/dx eˣ = eˣ = out.data
    out._backward = _backward
    return out
```

### `pow` (and Therefore `truediv`)

We implement raising to a **constant** power `k` (an int or float):

```python
def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out
```

Then division is just `a · b⁻¹`:

```python
def __truediv__(self, other):
    return self * other ** -1
```

### Negation and Subtraction

```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)
```

### Re-deriving `tanh`

```python
e  = (2 * n).exp()
o  = (e - 1) / (e + 1)
o.backward()
```

The forward value matches the old `tanh(n)`. The leaf gradients (on `x1, x2, w1, w2`) are **identical** — proving that *how* you decompose an operation doesn't change the final gradients, only the size of the graph.

> 🎯 **Big-picture lesson.** *The level of abstraction at which you implement a "node" is up to you.* Anything from a single `+` to an entire `LayerNorm` block is fine, as long as you can write its forward and its **local gradient**.

---

## 15. Same Thing in PyTorch

Karpathy modelled micrograd on PyTorch on purpose. Here's the *same* neuron:

```python
import torch

x1 = torch.Tensor([2.0]).double();    x1.requires_grad = True
x2 = torch.Tensor([0.0]).double();    x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double();   w1.requires_grad = True
w2 = torch.Tensor([1.0]).double();    w2.requires_grad = True
b  = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())   # 0.7071...
o.backward()

print(x1.grad.item(), x2.grad.item(), w1.grad.item(), w2.grad.item())
# -1.5    0.5    1.0    0.0
```

Identical gradients. PyTorch differs in two ways:

1. **Tensors, not scalars.** A tensor is an n-D array; `Tensor([2.0])` creates a 1-element tensor. `.item()` extracts the single Python number.
2. **`requires_grad=False` by default for leaves.** Real networks have huge inputs you don't want to differentiate w.r.t., so PyTorch is conservative for efficiency.

Otherwise the API surface is the same:

| Concept | micrograd | PyTorch |
|--------|-----------|---------|
| The wrapper | `Value` | `Tensor` |
| Forward number | `.data` | `.data` (tensor) / `.item()` (scalar) |
| Gradient | `.grad` | `.grad` |
| Run backprop | `.backward()` | `.backward()` |

> When you scale up to billions of parameters, **the maths is exactly the same**. PyTorch is just *fast*: parallelism over arrays, GPU kernels, and clever memory management.

---

## 16. Building a Neural Net Library

We have an autograd engine. Now stack three small classes on top: **`Neuron` → `Layer` → `MLP`**.

### `Neuron`

A neuron has `nin` weights (one per input) plus a bias, and an activation:

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w·x + b, then squash
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]
```

Notes:

- `__call__` lets us write `n(x)` like a function.
- `sum(generator, start=self.b)` is a tidy way to start the running sum *at the bias* instead of at zero.
- `zip(self.w, x)` pairs each weight with its input.

### `Layer`

A layer is just a list of independent neurons:

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
```

Returning a single `Value` (instead of a 1-element list) when the layer has only one neuron is a small convenience — useful for the *output* layer of a regressor/binary classifier.

### `MLP` — Multi-Layer Perceptron

A list of layers, applied sequentially:

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts                 # e.g. [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Constructing the network from the lecture:

```python
n = MLP(3, [4, 4, 1])    # 3 inputs → hidden 4 → hidden 4 → 1 output
x = [2.0, 3.0, -1.0]
n(x)                     # one forward pass — returns a single Value
```

Calling `draw_dot(n(x))` reveals an **enormous** computation graph. We will *never* derive its gradient on paper — but micrograd does it for us automatically.

> The `parameters()` method is mirrored on every container so we can later collect *all* the weights and biases of the entire MLP with `n.parameters()`.

---

## 17. A Tiny Dataset and the Loss Function

Tiny binary-classification dataset:

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]   # desired targets
```

Initial predictions:

```python
ypred = [n(x) for x in xs]
# e.g. [0.91, -0.88, 0.80, 0.80]   — pretty wrong
```

We need *one number* that summarises **how badly** the network is doing. That number is the **loss**.

### Mean Squared Error (MSE)

```python
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

Why squaring?

- It removes the sign (we don't care if the prediction is too high or too low — both are bad).
- It is `0` *iff* `yout == ygt` — a perfect prediction.
- It punishes large errors more than small ones.

A high loss means bad predictions. **We want to minimise the loss.**

### Backprop on the Loss

```python
loss.backward()
```

After this single line, **every parameter of every neuron in every layer** has a `.grad` telling us *how a tiny change in that parameter would change the loss*. We didn't write a single derivative — micrograd did it.

Inspecting one weight:

```python
n.layers[0].neurons[0].w[0].grad      # e.g. -0.41
```

Negative grad → *increasing* this weight slightly **decreases** the loss. The opposite for a positive grad. **That's all we need to train.**

> The graph that `loss.backward()` traverses is huge: it includes 4 forward passes (one per training example) plus the loss computation. Drawing it would fill a wall.

---

## 18. Gradient Descent — Training the Network

### One Update Step

```python
for p in n.parameters():
    p.data += -0.01 * p.grad      # ⚠ note the MINUS sign
```

- The gradient points in the direction of **increasing** loss.
- We want to **decrease** loss → step in the *opposite* direction.
- `0.01` is the **step size / learning rate**.

After the update, recompute the loss — it should be a little lower. Repeat.

### A Real Training Loop

```python
for k in range(20):
    # 1. forward pass
    ypred = [n(x) for x in xs]
    loss  = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 2. backward pass
    for p in n.parameters():
        p.grad = 0.0           # ← we'll explain this in §19
    loss.backward()

    # 3. update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
```

Watch the loss decrease step by step:

```
0   7.12
1   4.36
2   3.66
...
19  0.0008
```

After 20 steps the predictions are close to `[1, -1, -1, 1]` — *the network has learnt*.

### The Learning Rate Is A Subtle Art

- **Too small** → training crawls; thousands of steps to converge.
- **Too large** → you "overshoot" and the loss can *blow up*.
- Often: start big, decay to small near the end (**learning-rate decay**).

You only know the *local* shape of the loss landscape. Step too far and you may land in completely unfamiliar territory. *That's why people obsess over learning-rate schedules.*

---

## 19. The `zero_grad` Bug

Karpathy intentionally leaves a bug in the first training loop and then catches it:

> "I forgot to **zero_grad before backward**." — Tweet of *common neural-net mistakes*, item #3.

What happened? Recall that every `_backward()` does `self.grad += …`. It **accumulates**. So if we don't *reset* the gradients before each backward pass, gradients from the previous iteration leak into the next one. The "step direction" becomes nonsense. In a tiny problem like ours it accidentally worked, but in any real network it would be catastrophic.

### The Fix

Before every backward pass:

```python
for p in n.parameters():
    p.grad = 0.0
```

PyTorch has the same idiom: `optimizer.zero_grad()` (or `model.zero_grad()`).

### Why Does Backward *Accumulate*?

Because of the multi-use-variable issue from §13. Inside the graph, a parameter (or any value) can be reached via several paths. Their gradients **must add**. So `_backward` always does `+=`. Resetting to zero is the *user's* responsibility before each new pass.

> **Mantra.** `zero_grad → forward → backward → step`. In that exact order.

---

## 20. Summary — From Micrograd to Modern Neural Nets

We can now state the *whole* paradigm in one paragraph:

> A neural network is a **mathematical expression** parameterised by **weights** and fed by **data**. We define a **loss function** that is *low* exactly when the network does what we want. We compute the loss (forward pass), use **backpropagation** to obtain the gradient of the loss with respect to every weight, and **nudge** every weight a tiny step against its gradient (gradient descent). We iterate this process. As the loss falls, the network gets better.

Differences when scaling up to a state-of-the-art model (e.g. GPT):

| What we did | What real models use |
|-------------|----------------------|
| Scalar `Value` | n-dimensional `Tensor` (GPU-parallel) |
| `tanh` | Often `ReLU`, `GeLU`, `SwiGLU` … |
| Mean Squared Error | **Cross-entropy** for classification / next-token prediction |
| Vanilla SGD | **Adam, AdamW** (with momentum, adaptive learning rates) |
| 41 parameters | hundreds of billions to trillions |

But the *algorithm* — forward, backward, step — is **identical**. That's the great unification of the lecture.

---

## 21. Walkthrough of the Real Micrograd Repo

The repo contains essentially two source files. Below is a faithful version of each.

### 21.1 `engine.py` — The Whole Autograd Engine

```python
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    # convenience wrappers built on the four primitives above
    def __neg__(self):           return self * -1
    def __radd__(self, other):   return self + other
    def __sub__(self, other):    return self + (-other)
    def __rsub__(self, other):   return other + (-self)
    def __rmul__(self, other):   return self * other
    def __truediv__(self, other):  return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

**~100 lines. That's the whole engine.** Notice:

- The `relu` non-linearity here replaces `tanh`. Its local derivative is `1` if `out > 0` else `0` — written compactly as `(out.data > 0) * out.grad`.
- All operators (`+, *, **`) accept either another `Value` or a plain Python number.
- `backward()` does the topological sort and walks the graph in reverse — exactly as we built it in §12.

### 21.2 `nn.py` — The Whole Neural-Net Library

```python
import random
from micrograd.engine import Value

class Module:
    """ Base class — mirrors torch.nn.Module """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # the last layer is *linear* (no non-linearity) so we can output any real number
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1))
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

This is `nn.py`. `Module` mirrors PyTorch's `nn.Module` and even has the same `zero_grad()` method. The **last layer is linear** (no non-linearity) so the network can output any real number — a small but important detail.

### 21.3 The `demo.ipynb` Notebook

The repo's demo is *slightly* fancier than what we built:

- A bigger 2-D dataset (clouds of red and blue points).
- A larger MLP.
- **Mini-batching** — for big datasets you don't forward the whole set every step; you randomly sample a batch.
- **Max-margin (hinge) loss** — another option for binary classification.
- **L2 regularisation** — adds `λ · Σ wᵢ²` to the loss to discourage huge weights and reduce overfitting.
- **Learning-rate decay** — the LR shrinks across iterations.

These details are not covered in the lecture but are standard tools in the deep-learning toolbox.

---

## 22. Glimpse Into Production — PyTorch's `tanh` Backward

Karpathy goes on a brief safari into the PyTorch source to find the production code for `tanh.backward`. The result is humbling:

- Searching "tanh" gives **2,800+ matches across 400+ files**. Real libraries are *huge*.
- After much hunting, the actual CPU kernel for `tanh.backward` lives in `BinaryOpsKernel.cpp` (despite `tanh` not being a binary op…).
- Buried inside, separated from boilerplate for complex types, half-precision floats, etc., is essentially:

  ```cpp
  // a is grad_output, b is the saved tanh output
  return a * (Scalar(1) - b * b);
  ```

  i.e. `out.grad * (1 - tanh²(x))`. **Exactly what we wrote in micrograd.** 🎉

The GPU (CUDA) version is a single line on the device.

### Adding A Custom Op To PyTorch

PyTorch lets you define new differentiable operations by sub-classing `torch.autograd.Function` and implementing `forward` and `backward`. Here's the structure (Legendre polynomial 3 from PyTorch docs):

```python
class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input**3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input**2 - 1)
```

> The contract is **exactly** what `_backward` is in micrograd: *given the upstream gradient and the saved forward inputs/output, return the gradient with respect to each input.*

Once you've done this, your op becomes a Lego brick that PyTorch can compose into any larger computation.

---

## 23. Final Cheat Sheet

### The `Value` Mental Model

```
data:      the scalar number from the forward pass
grad:      ∂(loss) / ∂(this node)  — filled in by backward()
_prev:     the parent Value(s) that produced this node
_op:       the operation that produced this node (for debugging / drawing)
_backward: a closure that pushes my .grad into my parents' .grad
```

### Forward Operation → Local Derivative Cheat Sheet

| Operation | Forward (`out = f(a, b, …)`) | Local derivative w.r.t. each input |
|-----------|------------------------------|-----------------------------------|
| Add | `a + b` | `1`, `1` |
| Mul | `a * b` | `b`, `a` |
| Sub | `a - b` | `1`, `-1` (built from `+` and `-`) |
| Pow (constant) | `a ** k` | `k · a^(k-1)` |
| Div | `a / b` = `a · b⁻¹` | from `*` and `**-1` |
| Exp | `eᵃ` | `eᵃ` (i.e. `out`) |
| `tanh` | `tanh(a)` | `1 - tanh²(a)` |
| `ReLU` | `max(0, a)` | `1` if `a > 0` else `0` |

### The Universal Training Loop

```python
for step in range(num_iters):
    ypred = model(xs)
    loss  = loss_fn(ypred, ys)

    for p in model.parameters():
        p.grad = 0.0           # zero_grad

    loss.backward()            # chain rule through the whole graph

    for p in model.parameters():
        p.data += -lr * p.grad # step in the OPPOSITE direction of the gradient
```

### Common Pitfalls (Karpathy's "Most Common Neural Net Mistakes")

- ❌ **Forgetting `zero_grad`** before `backward()` (see §19).
- ❌ **`out._backward = _backward()`** instead of `_backward` (calling vs. storing).
- ❌ Setting gradients with `=` instead of `+=` (breaks multi-use variables).
- ❌ Learning rate too big → loss explodes. Too small → never converges.
- ❌ Gradient checking with `h` too small → floating-point garbage.

### Three Sentences That Summarise the Whole Lecture

1. A neural network is just a mathematical expression with parameters.
2. Backpropagation is the recursive application of the chain rule, which gives us the gradient of the loss with respect to every parameter.
3. Gradient descent then moves every parameter a tiny step *against* its gradient, lowering the loss — and after enough steps the network learns to do what we want.

---

> **End of notes for Lecture 1 — *The Spelled-Out Intro to Neural Networks and Backpropagation: Building Micrograd*.**
>
> *"Micrograd is what you need to train neural networks. Everything else is just efficiency."* — Andrej Karpathy
