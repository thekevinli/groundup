import numpy as np

rng = np.random.default_rng(0)


def num_grad(f, x: np.ndarray, dout: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # numerical gradient of sum(f(x) * dout) w.r.t. x by central differences (x is changed in place)
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        hi = np.sum(f(x) * dout)
        x[i] = old - eps
        lo = np.sum(f(x) * dout)
        x[i] = old
        grad[i] = (hi - lo) / (2 * eps)
    return grad


# ---------------- affine: out = x @ w + b, with w of shape (D, M)
def affine_forward(x, w, b):
    out = x @ w + b                                    # any leading dims: (..., D) -> (..., M)
    cache = (x, w)
    return out, cache


def affine_backward(dout, cache):
    x, w = cache
    x2, dout2 = x.reshape(-1, x.shape[-1]), dout.reshape(-1, dout.shape[-1])   # every position used w
    dx = dout @ w.T
    dw = x2.T @ dout2
    db = dout2.sum(axis=0)
    return dx, dw, db


# ---------------- activations: elementwise, so dx = dout * f'(x)
def relu_forward(x):
    return np.maximum(x, 0), x


def relu_backward(dout, cache):
    x = cache
    return dout * (x > 0)


def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    return out, out


def sigmoid_backward(dout, cache):
    out = cache                                        # sigma' = sigma (1 - sigma): reuse the output
    return dout * out * (1 - out)


def gelu_forward(x):
    # tanh approximation (GPT-2): out = 0.5 x (1 + tanh(u)), u = c (x + 0.044715 x^3)
    c = np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 + t), (x, t)


def gelu_backward(dout, cache):
    x, t = cache
    du = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    return dout * (0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du)        # product rule


# ---------------- softmax over the last dim: dx = p * (dout - sum(dout * p)), no S x S Jacobian
def softmax_forward(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))      # subtract the max: no overflow
    p = e / e.sum(axis=-1, keepdims=True)
    return p, p


def softmax_backward(dout, cache):
    p = cache
    return p * (dout - (dout * p).sum(axis=-1, keepdims=True))


# ---------------- losses: return the loss and its gradient together
def mse_loss(pred, target):
    diff = pred - target
    loss = np.mean(diff ** 2)
    dpred = 2 * diff / diff.size
    return loss, dpred


def softmax_loss(z, y):
    # z: (N, C) logits, y: (N,) labels. Softmax + log cancel: dz = (p - onehot(y)) / N
    N = z.shape[0]
    p, _ = softmax_forward(z)
    loss = -np.mean(np.log(p[np.arange(N), y]))
    dz = p.copy()
    dz[np.arange(N), y] -= 1
    return loss, dz / N


# ---------------- layernorm over features (axis -1); batchnorm is the same over the batch (axis 0)
def layernorm_forward(x, gamma, beta, eps=1e-5, axis=-1):
    mu = x.mean(axis=axis, keepdims=True)
    std = np.sqrt(x.var(axis=axis, keepdims=True) + eps)
    xh = (x - mu) / std
    out = gamma * xh + beta
    return out, (xh, std, gamma, axis)


def layernorm_backward(dout, cache):
    xh, std, gamma, axis = cache
    other = tuple(range(dout.ndim - 1))                # gamma, beta are per feature: sum over the rest
    dgamma = (dout * xh).sum(axis=other)
    dbeta = dout.sum(axis=other)
    g = dout * gamma
    dx = (g - g.mean(axis=axis, keepdims=True) - xh * (g * xh).mean(axis=axis, keepdims=True)) / std
    return dx, dgamma, dbeta


def batchnorm_forward(x, gamma, beta, eps=1e-5):
    return layernorm_forward(x, gamma, beta, eps, axis=0)   # x: (N, D), statistics per feature


def batchnorm_backward(dout, cache):
    return layernorm_backward(dout, cache)


# ---------------- causal multi-head self-attention; params: wq, bq, wk, bk, wv, bv, wo, bo
def split_heads(t, H):                                 # (B, S, D) -> (B, H, S, D/H)
    B, S, D = t.shape
    return t.reshape(B, S, H, D // H).transpose(0, 2, 1, 3)


def merge_heads(t):                                    # (B, H, S, hd) -> (B, S, H * hd)
    B, H, S, hd = t.shape
    return t.transpose(0, 2, 1, 3).reshape(B, S, H * hd)


def attention_forward(x, params, H):
    q, cq = affine_forward(x, params["wq"], params["bq"])
    k, ck = affine_forward(x, params["wk"], params["bk"])
    v, cv = affine_forward(x, params["wv"], params["bv"])
    q, k, v = split_heads(q, H), split_heads(k, H), split_heads(v, H)
    S, scale = x.shape[1], q.shape[-1] ** -0.5
    s = q @ k.transpose(0, 1, 3, 2) * scale
    s = np.where(np.tril(np.ones((S, S), dtype=bool)), s, -np.inf)    # causal: exp(-inf) = 0
    p, cp = softmax_forward(s)
    out, co = affine_forward(merge_heads(p @ v), params["wo"], params["bo"])
    return out, (cq, ck, cv, cp, co, q, k, v, p, scale, H)


def attention_backward(dout, cache):
    cq, ck, cv, cp, co, q, k, v, p, scale, H = cache
    grads = {}
    da, grads["wo"], grads["bo"] = affine_backward(dout, co)
    da = split_heads(da, H)
    dp = da @ v.transpose(0, 1, 3, 2)                  # a = p v
    dv = p.transpose(0, 1, 3, 2) @ da
    ds = softmax_backward(dp, cp) * scale              # masked entries: p = 0, so ds = 0
    dq = ds @ k                                        # s = q k^T
    dk = ds.transpose(0, 1, 3, 2) @ q
    dxq, grads["wq"], grads["bq"] = affine_backward(merge_heads(dq), cq)
    dxk, grads["wk"], grads["bk"] = affine_backward(merge_heads(dk), ck)
    dxv, grads["wv"], grads["bv"] = affine_backward(merge_heads(dv), cv)
    return dxq + dxk + dxv, grads                      # x fed all three projections: add the paths


# ---------------- pre-norm block: h = x + attn(ln1(x)), y = h + fc2(gelu(fc1(ln2(h))))
def block_forward(x, params, H):
    a, c1 = layernorm_forward(x, params["g1"], params["b1"])
    a, ca = attention_forward(a, params["attn"], H)
    h = x + a
    m, c2 = layernorm_forward(h, params["g2"], params["b2"])
    m, cf1 = affine_forward(m, params["w1"], params["c1"])
    m, cg = gelu_forward(m)
    m, cf2 = affine_forward(m, params["w2"], params["c2"])
    return h + m, (c1, ca, c2, cf1, cg, cf2)


def block_backward(dout, cache):
    # every forward step in reverse; a residual add copies the gradient to both branches
    c1, ca, c2, cf1, cg, cf2 = cache
    grads = {}
    dm, grads["w2"], grads["c2"] = affine_backward(dout, cf2)
    dm = gelu_backward(dm, cg)
    dm, grads["w1"], grads["c1"] = affine_backward(dm, cf1)
    dm, grads["g2"], grads["b2"] = layernorm_backward(dm, c2)
    dh = dout + dm
    da, grads["attn"] = attention_backward(dh, ca)
    da, grads["g1"], grads["b1"] = layernorm_backward(da, c1)
    return dh + da, grads


def test_affine():
    x, w, b = rng.normal(size=(2, 5, 4)), rng.normal(size=(4, 3)), rng.normal(size=3)
    out, cache = affine_forward(x, w, b)
    dout = rng.normal(size=out.shape)
    dx, dw, db = affine_backward(dout, cache)
    assert np.allclose(dx, num_grad(lambda x: affine_forward(x, w, b)[0], x, dout), atol=1e-5)
    assert np.allclose(dw, num_grad(lambda w: affine_forward(x, w, b)[0], w, dout), atol=1e-5)
    assert np.allclose(db, num_grad(lambda b: affine_forward(x, w, b)[0], b, dout), atol=1e-5)


def test_activations():
    x = rng.normal(size=(3, 4)) + 0.01                 # + 0.01: keep ReLU's kink off zero
    dout = rng.normal(size=x.shape)
    for fwd, bwd in ((relu_forward, relu_backward), (sigmoid_forward, sigmoid_backward), (gelu_forward, gelu_backward)):
        dx = bwd(dout, fwd(x)[1])
        assert np.allclose(dx, num_grad(lambda x: fwd(x)[0], x, dout), atol=1e-5), fwd.__name__


def test_softmax():
    x = rng.normal(size=(2, 3, 5))
    dout = rng.normal(size=x.shape)
    dx = softmax_backward(dout, softmax_forward(x)[1])
    assert np.allclose(dx, num_grad(lambda x: softmax_forward(x)[0], x, dout), atol=1e-5)


def test_losses():
    p, t = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    _, dp = mse_loss(p, t)
    assert np.allclose(dp, num_grad(lambda p: mse_loss(p, t)[0], p, np.array(1.0)), atol=1e-6)
    z, y = rng.normal(size=(5, 7)), rng.integers(0, 7, 5)
    _, dz = softmax_loss(z, y)
    assert np.allclose(dz, num_grad(lambda z: softmax_loss(z, y)[0], z, np.array(1.0)), atol=1e-6)


def test_norms():
    for fwd, bwd, shape in ((layernorm_forward, layernorm_backward, (2, 3, 6)), (batchnorm_forward, batchnorm_backward, (8, 6))):
        x, gamma, beta = rng.normal(size=shape), rng.normal(size=6), rng.normal(size=6)
        out, cache = fwd(x, gamma, beta)
        dout = rng.normal(size=out.shape)
        dx, dgamma, dbeta = bwd(dout, cache)
        assert np.allclose(dx, num_grad(lambda x: fwd(x, gamma, beta)[0], x, dout), atol=1e-5)
        assert np.allclose(dgamma, num_grad(lambda g: fwd(x, g, beta)[0], gamma, dout), atol=1e-5)
        assert np.allclose(dbeta, num_grad(lambda b: fwd(x, gamma, b)[0], beta, dout), atol=1e-5)


def attention_params(D):
    return {n: rng.normal(0, D ** -0.5, (D, D)) if n[0] == "w" else rng.normal(size=D)
            for n in ("wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo")}


def test_attention():
    x, params = rng.normal(size=(2, 5, 8)), attention_params(8)
    out, cache = attention_forward(x, params, 2)
    dout = rng.normal(size=out.shape)
    dx, grads = attention_backward(dout, cache)
    assert np.allclose(dx, num_grad(lambda x: attention_forward(x, params, 2)[0], x, dout), atol=1e-5)
    for n, p in params.items():
        assert np.allclose(grads[n], num_grad(lambda _: attention_forward(x, params, 2)[0], p, dout), atol=1e-5), n


def test_block():
    D = 8
    params = {"g1": rng.normal(size=D), "b1": rng.normal(size=D), "attn": attention_params(D),
              "g2": rng.normal(size=D), "b2": rng.normal(size=D),
              "w1": rng.normal(0, D ** -0.5, (D, 4 * D)), "c1": rng.normal(size=4 * D),
              "w2": rng.normal(0, (4 * D) ** -0.5, (4 * D, D)), "c2": rng.normal(size=D)}
    x = rng.normal(size=(2, 5, D))
    out, cache = block_forward(x, params, 2)
    dout = rng.normal(size=out.shape)
    dx, grads = block_backward(dout, cache)
    f = lambda _: block_forward(x, params, 2)[0]
    assert np.allclose(dx, num_grad(lambda x: block_forward(x, params, 2)[0], x, dout), atol=1e-5)
    for n in ("g1", "w1", "c2", "g2"):
        assert np.allclose(grads[n], num_grad(f, params[n], dout), atol=1e-5), n
    assert np.allclose(grads["attn"]["wq"], num_grad(f, params["attn"]["wq"], dout), atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: analytic gradients match finite differences")
