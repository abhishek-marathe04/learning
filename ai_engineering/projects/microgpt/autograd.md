# Autograd Engine — `Value` Class Internals

> **Tags:** `#autograd` `#backprop` `#value` `#gradients` `#chain-rule`

---

## `__slots__` on the `Value` class

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
```

Normally every Python object carries a `__dict__` (a full dictionary) to store its attributes — flexible but memory-heavy (~200–400 bytes per object). `__slots__` tells Python: **this class will only ever have exactly these four attributes**. Python then allocates fixed-size slots instead of a dict, saving ~3–5x memory per object.

This matters in microgpt because the computation graph creates **tens of thousands of `Value` objects** — every scalar weight, every intermediate result. Without `__slots__`, memory usage would balloon significantly.

The four slots map exactly to what autograd needs:
- `data` — the scalar value computed in the forward pass
- `grad` — dL/d(this node), filled in by `backward()`
- `_children` — inputs to this operation (who created me?)
- `_local_grads` — ∂(this node)/∂(each child) — the local derivative

---

## Where derivatives are computed — at operation time

This is the key insight of the `Value` class. Derivatives are **not** computed during `backward()`. They are computed and stored **at the moment each operation executes**:

```python
def __add__(self, other):
    return Value(self.data + other.data, (self, other), (1, 1))
#                                                        ↑ ↑
#                                         ∂(a+b)/∂a=1   ∂(a+b)/∂b=1

def __mul__(self, other):
    return Value(self.data * other.data, (self, other), (other.data, self.data))
#                                                        ↑            ↑
#                                         ∂(a*b)/∂a=b   ∂(a*b)/∂b=a

def __pow__(self, other):
    return Value(self.data**other, (self,), (other * self.data**(other-1),))
#                                            ↑ power rule: n·aⁿ⁻¹

def log(self):
    return Value(math.log(self.data), (self,), (1/self.data,))
#                                               ↑ ∂ln(a)/∂a = 1/a

def exp(self):
    return Value(math.exp(self.data), (self,), (math.exp(self.data),))
#                                               ↑ ∂eˣ/∂x = eˣ

def relu(self):
    return Value(max(0, self.data), (self,), (float(self.data > 0),))
#                                             ↑ 1 if positive, 0 if negative
```

**Full table of local gradients:**

| Operation | Forward | Local gradient |
|-----------|---------|----------------|
| `a + b` | a + b | ∂/∂a = 1, ∂/∂b = 1 |
| `a * b` | a · b | ∂/∂a = b, ∂/∂b = a |
| `a ** n` | aⁿ | ∂/∂a = n·aⁿ⁻¹ |
| `log(a)` | ln(a) | ∂/∂a = 1/a |
| `exp(a)` | eᵃ | ∂/∂a = eᵃ |
| `relu(a)` | max(0,a) | ∂/∂a = 1 if a>0 else 0 |

---

## `backward()` — just a messenger

`backward()` is **not** where math happens. It walks the graph in reverse topological order and multiplies stored local grads by the flowing gradient:

```python
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad   # chain rule: just multiplication
```

The `+=` (accumulation) handles the case where a node is used in multiple places — gradients from all paths must be summed.

**Mental model:**
- Forward pass: each op computes its output AND stores its local derivative right then
- Backward pass: gradients flow backward, each node multiplies the incoming gradient by its stored local grad and passes it to its children
- The chain rule emerges from this multiplication cascade — no calculus required at backward time
