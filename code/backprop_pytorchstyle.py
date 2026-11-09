import numpy as np

rng = np.random.default_rng(0)


def num_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # central differences of the scalar f() with respect to every entry of x (changed in place)
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        hi = f()
        x[i] = old - eps
        lo = f()
        x[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def check(layer, *inputs):
    # loss = sum(y * G) for a random G, so dL/dy = G. Analytic grads first, then numeric ones.
    G = rng.normal(size=np.shape(layer.forward(*inputs)))
    dxs = layer.backward(G)
    dxs = dxs if isinstance(dxs, tuple) else (dxs,)
    loss = lambda: np.sum(layer.forward(*inputs) * G)
    for x, dx in zip(inputs, dxs):
        assert np.allclose(dx, num_grad(loss, x), atol=1e-5), type(layer).__name__
    for name, p in layer.params().items():
        assert np.allclose(layer.grads[name], num_grad(loss, p), atol=1e-5), name


class Linear:
    # y = x W^T + b. Each output is a dot product, so: dx = dy W, dW = dy^T x, db = sum of dy.
    def __init__(self, d_in: int, d_out: int):
        self.W, self.b = rng.normal(0, d_in ** -0.5, (d_out, d_in)), np.zeros(d_out)

    def params(self):
        return {"W": self.W, "b": self.b}

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dy):
        x2, dy2 = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])   # any leading dims
        self.grads = {"W": dy2.T @ x2, "b": dy2.sum(0)}         # sum over every position that used W
        return dy @ self.W


class ReLU:
    # y = max(x, 0): the gradient passes where x > 0 and is cut where x < 0
    def params(self):
        return {}

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dy):
        return dy * (self.x > 0)


class Sigmoid:
    # y = 1 / (1 + e^-x), dy/dx = y (1 - y): reuse the output, at most 1/4, so it vanishes in deep stacks
    def params(self):
        return {}

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        return dy * self.y * (1 - self.y)


class GELU:
    # tanh approximation, as in GPT-2: y = x * s(x), s = 0.5 (1 + tanh(u)), u = c (x + 0.044715 x^3)
    def params(self):
        return {}

    def forward(self, x):
        c = np.sqrt(2 / np.pi)
        self.x, self.t = x, np.tanh(c * (x + 0.044715 * x ** 3))
        return 0.5 * x * (1 + self.t)

    def backward(self, dy):
        x, t, c = self.x, self.t, np.sqrt(2 / np.pi)
        du = c * (1 + 3 * 0.044715 * x ** 2)                         # product rule: s(x) + x s'(x)
        return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du)


class Softmax:
    # y_i = e^x_i / sum_j e^x_j along the last dim. Jacobian dy_i/dx_j = y_i (delta_ij - y_j), so
    # dx = y * (dy - sum(dy * y)): never build the S x S Jacobian
    def params(self):
        return {}

    def forward(self, x):
        e = np.exp(x - x.max(-1, keepdims=True))                     # subtract the max: no overflow
        self.y = e / e.sum(-1, keepdims=True)
        return self.y

    def backward(self, dy):
        return self.y * (dy - (dy * self.y).sum(-1, keepdims=True))


class MSE:
    # L = mean((p - t)^2) over all N entries: dL/dp = 2 (p - t) / N
    def forward(self, p, t):
        self.d = p - t
        return np.mean(self.d ** 2)

    def backward(self):
        return 2 * self.d / self.d.size


class CrossEntropy:
    # logits z (N, V), integer targets (N,). L = mean(logsumexp(z) - z[target]).
    # Softmax and log cancel: dL/dz = (softmax(z) - onehot(target)) / N. Never backprop through log(p).
    def forward(self, z, target):
        z = z - z.max(-1, keepdims=True)
        self.p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        self.target = target
        return -np.mean(np.log(self.p[np.arange(len(target)), target]))

    def backward(self):
        dz = self.p.copy()
        dz[np.arange(len(self.target)), self.target] -= 1
        return dz / len(self.target)


class LayerNorm:
    # normalize each token over its features (axis -1). With xh = (x - mean) / std, g = dy * gamma:
    #   dx = (g - mean(g) - xh * mean(g * xh)) / std        (means over the normalized axis)
    # BatchNorm is the same formula over the batch axis instead (training mode).
    axis = -1

    def __init__(self, d: int, eps: float = 1e-5):
        self.gamma, self.beta, self.eps = np.ones(d), np.zeros(d), eps

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def forward(self, x):
        mu = x.mean(self.axis, keepdims=True)
        self.std = np.sqrt(x.var(self.axis, keepdims=True) + self.eps)
        self.xh = (x - mu) / self.std
        return self.gamma * self.xh + self.beta

    def backward(self, dy):
        other = tuple(range(dy.ndim - 1))                             # gamma, beta: sum over all but features
        self.grads = {"gamma": (dy * self.xh).sum(other), "beta": dy.sum(other)}
        g, a = dy * self.gamma, self.axis
        return (g - g.mean(a, keepdims=True) - self.xh * (g * self.xh).mean(a, keepdims=True)) / self.std


class BatchNorm(LayerNorm):
    # x: (N, d): statistics per feature over the batch. Inference uses running averages instead.
    axis = 0


class Attention:
    # causal multi-head self-attention, built from the pieces above
    def __init__(self, d: int, heads: int):
        self.h = heads
        self.q, self.k, self.v, self.o = (Linear(d, d) for _ in range(4))
        self.softmax = Softmax()

    def params(self):
        return {f"{n}.{k}": p for n in "qkvo" for k, p in getattr(self, n).params().items()}

    @property
    def grads(self):
        return {f"{n}.{k}": g for n in "qkvo" for k, g in getattr(self, n).grads.items()}

    def split(self, t):                                               # (B, S, d) -> (B, H, S, hd)
        B, S, d = t.shape
        return t.reshape(B, S, self.h, d // self.h).transpose(0, 2, 1, 3)

    def merge(self, t):                                               # (B, H, S, hd) -> (B, S, d)
        B, H, S, hd = t.shape
        return t.transpose(0, 2, 1, 3).reshape(B, S, H * hd)

    def forward(self, x):
        q, k, v = (self.split(l.forward(x)) for l in (self.q, self.k, self.v))
        S, self.scale = x.shape[1], q.shape[-1] ** -0.5
        s = q @ k.transpose(0, 1, 3, 2) * self.scale
        s = np.where(np.tril(np.ones((S, S), bool)), s, -np.inf)       # causal mask: exp(-inf) = 0
        self.qkv, self.p = (q, k, v), self.softmax.forward(s)
        return self.o.forward(self.merge(self.p @ v))

    def backward(self, dy):
        q, k, v = self.qkv
        da = self.split(self.o.backward(dy))                          # d(attention output)
        dp, dv = da @ v.transpose(0, 1, 3, 2), self.p.transpose(0, 1, 3, 2) @ da   # a = p v
        ds = self.softmax.backward(dp) * self.scale                   # masked entries: p = 0, so ds = 0
        dq, dk = ds @ k, ds.transpose(0, 1, 3, 2) @ q                 # s = q k^T
        # x fed all three projections: its gradient is the sum of the three paths
        return sum(l.backward(self.merge(d)) for l, d in ((self.q, dq), (self.k, dk), (self.v, dv)))


class Block:
    # pre-norm transformer block: h = x + attn(ln1(x)), y = h + mlp(ln2(h)).
    # A residual add copies the gradient to both branches: dh = dy + (gradient through the MLP).
    def __init__(self, d: int, heads: int):
        self.ln1, self.attn, self.ln2 = LayerNorm(d), Attention(d, heads), LayerNorm(d)
        self.fc1, self.act, self.fc2 = Linear(d, 4 * d), GELU(), Linear(4 * d, d)
        self.parts = {"ln1": self.ln1, "attn": self.attn, "ln2": self.ln2, "fc1": self.fc1, "fc2": self.fc2}

    def params(self):
        return {f"{n}.{k}": p for n, m in self.parts.items() for k, p in m.params().items()}

    @property
    def grads(self):
        return {f"{n}.{k}": g for n, m in self.parts.items() for k, g in m.grads.items()}

    def forward(self, x):
        h = x + self.attn.forward(self.ln1.forward(x))
        return h + self.fc2.forward(self.act.forward(self.fc1.forward(self.ln2.forward(h))))

    def backward(self, dy):                                           # every forward step, in reverse
        dh = dy + self.ln2.backward(self.fc1.backward(self.act.backward(self.fc2.backward(dy))))
        return dh + self.ln1.backward(self.attn.backward(dh))


def test_linear():
    check(Linear(4, 3), rng.normal(size=(2, 5, 4)))


def test_activations():
    for act in (ReLU(), Sigmoid(), GELU()):
        check(act, rng.normal(size=(3, 4)) + 0.01)                  # + 0.01: keep ReLU's kink off zero


def test_softmax():
    check(Softmax(), rng.normal(size=(2, 3, 5)))


def test_losses():
    p, t = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    mse = MSE()
    mse.forward(p, t)
    assert np.allclose(mse.backward(), num_grad(lambda: mse.forward(p, t), p), atol=1e-6)
    z, target = rng.normal(size=(5, 7)), rng.integers(0, 7, 5)
    ce = CrossEntropy()
    ce.forward(z, target)
    assert np.allclose(ce.backward(), num_grad(lambda: ce.forward(z, target), z), atol=1e-6)


def test_norms():
    ln = LayerNorm(6)
    ln.gamma, ln.beta = rng.normal(size=6), rng.normal(size=6)
    check(ln, rng.normal(size=(2, 3, 6)))
    bn = BatchNorm(6)
    bn.gamma, bn.beta = rng.normal(size=6), rng.normal(size=6)
    check(bn, rng.normal(size=(8, 6)))


def test_attention():
    check(Attention(8, 2), rng.normal(size=(2, 5, 8)))


def test_block():
    block = Block(8, 2)
    for ln in (block.ln1, block.ln2):
        ln.gamma, ln.beta = rng.normal(size=8), rng.normal(size=8)
    check(block, rng.normal(size=(2, 5, 8)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: analytic gradients match finite differences")
