import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


class Activation:
    """Activation interface."""
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray, grad_a: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Identity(Activation):
    """Identity activation."""
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, z: np.ndarray, grad_a: np.ndarray) -> np.ndarray:
        return grad_a


class Sigmoid(Activation):
    """Sigmoid activation."""
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_clip = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def backward(self, z: np.ndarray, grad_a: np.ndarray) -> np.ndarray:
        a = self.forward(z)
        return grad_a * a * (1.0 - a)


class ReLU(Activation):
    """ReLU activation."""
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def backward(self, z: np.ndarray, grad_a: np.ndarray) -> np.ndarray:
        return grad_a * (z > 0.0)


class Softmax(Activation):
    """Softmax activation (kept for completeness; not used neuron-level)."""
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_shift = z - np.max(z, axis=1, keepdims=True)
        expz = np.exp(z_shift)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def backward(self, z: np.ndarray, grad_a: np.ndarray) -> np.ndarray:
        s = self.forward(z)
        batch, c = s.shape
        grad_z = np.zeros_like(s)
        for i in range(batch):
            si = s[i].reshape(-1, 1)
            j = np.diagflat(si) - si @ si.T
            grad_z[i] = (j @ grad_a[i].reshape(-1, 1)).ravel()
        return grad_z


class Loss:
    """Loss interface."""
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error loss."""
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_pred.shape[0]
        return (2.0 * (y_pred - y_true)) / n


class CrossEntropy(Loss):
    """Cross-entropy loss for one-hot labels."""
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        n = y_pred.shape[0]
        return -(y_true / y_pred) / n


@dataclass
class NeuronCache:
    """Cached values for a neuron."""
    x: np.ndarray
    z: np.ndarray
    a: np.ndarray


class Neuron:
    """A single neuron with weights, bias, and activation."""
    def __init__(self, in_dim: int, activation: Activation, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_dim) if isinstance(activation, ReLU) else np.sqrt(1.0 / in_dim)
        self.w = rng.normal(0.0, scale, size=(in_dim,))
        self.b = 0.0
        self.activation = activation
        self.dw = np.zeros_like(self.w)
        self.db = 0.0
        self.cache: Optional[NeuronCache] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w + self.b
        a = self.activation.forward(z)
        self.cache = NeuronCache(x=x, z=z, a=a)
        return a

    def backward(self, grad_a: np.ndarray) -> np.ndarray:
        assert self.cache is not None, "Call forward() before backward()."
        x, z = self.cache.x, self.cache.z
        grad_z = self.activation.backward(z, grad_a)
        self.dw = x.T @ grad_z
        self.db = float(np.sum(grad_z))
        return grad_z[:, None] * self.w[None, :]

    def step(self, lr: float, batch_size: int, weight_decay: float = 0.0):
        self.w -= lr * (self.dw / batch_size + weight_decay * self.w)
        self.b -= lr * (self.db / batch_size)


class Layer:
    """A dense layer implemented as a list of neurons."""
    def __init__(self, in_dim: int, out_dim: int, activation: Activation, rng: np.random.Generator):
        if isinstance(activation, Softmax):
            raise ValueError("Softmax cannot be neuron-level. Use Identity() + output_softmax=True.")
        self.neurons: List[Neuron] = [Neuron(in_dim, activation, rng) for _ in range(out_dim)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        outs = [n.forward(x) for n in self.neurons]
        return np.stack(outs, axis=1)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grads = [n.backward(grad_out[:, j]) for j, n in enumerate(self.neurons)]
        return np.sum(grads, axis=0)

    def step(self, lr: float, batch_size: int, weight_decay: float = 0.0):
        for n in self.neurons:
            n.step(lr, batch_size, weight_decay=weight_decay)


class NeuralNetwork:
    """A multi-layer perceptron built from neuron-based dense layers."""
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Activation],
        loss: Loss,
        seed: int = 42,
        output_softmax: bool = False,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output sizes.")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Provide exactly one activation per layer (excluding input).")

        self.rng = np.random.default_rng(seed)
        self.layers: List[Layer] = [
            Layer(layer_sizes[i], layer_sizes[i + 1], activations[i], self.rng)
            for i in range(len(layer_sizes) - 1)
        ]
        self.loss = loss
        self.output_softmax = output_softmax
        self._last_logits: Optional[np.ndarray] = None

    @staticmethod
    def _softmax_stable(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits, axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        self._last_logits = a
        return self._softmax_stable(a) if self.output_softmax else a

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss.forward(y_pred, y_true)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        if self.output_softmax and isinstance(self.loss, CrossEntropy):
            n = y_pred.shape[0]
            grad = (y_pred - y_true) / n
        else:
            grad = self.loss.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr: float, batch_size: int, weight_decay: float = 0.0) -> None:
        for layer in self.layers:
            layer.step(lr, batch_size, weight_decay=weight_decay)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        lr: float = 0.1,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        verbose_every: int = 1,
        shuffle: bool = True,
    ) -> List[float]:
        n = X.shape[0]
        losses: List[float] = []
        for epoch in range(1, epochs + 1):
            idx = np.arange(n)
            if shuffle:
                self.rng.shuffle(idx)
            Xs, ys = X[idx], y[idx]

            epoch_loss_accum = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb, yb = Xs[start:end], ys[start:end]
                y_pred = self.forward(xb)
                epoch_loss_accum += self.compute_loss(y_pred, yb)
                batches += 1
                self.backward(y_pred, yb)
                self.step(lr=lr, batch_size=xb.shape[0], weight_decay=weight_decay)

            epoch_loss = epoch_loss_accum / max(batches, 1)
            losses.append(epoch_loss)
            if verbose_every and (epoch % verbose_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:5d} | loss = {epoch_loss:.6f}")
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.forward(X)
        if y_pred.shape[1] == 1:
            return (y_pred[:, 0] >= 0.5).astype(int)
        return np.argmax(y_pred, axis=1)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot."""
    oh = np.zeros((y.shape[0], num_classes), dtype=float)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def evaluate(net: NeuralNetwork, X: np.ndarray, y_onehot: np.ndarray, batch_size: int = 256) -> float:
    """Compute accuracy over a dataset in batches."""
    n = X.shape[0]
    correct = 0
    total = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        probs = net.predict(X[start:end])
        pred = np.argmax(probs, axis=1)
        true = np.argmax(y_onehot[start:end], axis=1)
        correct += int(np.sum(pred == true))
        total += (end - start)
    return correct / total


def load_mnist_openml(
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from OpenML (requires scikit-learn)."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        raise ImportError("Install scikit-learn with: pip install scikit-learn") from e

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(np.int64)

    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    if limit_train is not None:
        X_train, y_train = X_train[:limit_train], y_train[:limit_train]
    if limit_test is not None:
        X_test, y_test = X_test[:limit_test], y_test[:limit_test]

    y_train_oh = one_hot(y_train, 10).astype(np.float32)
    y_test_oh = one_hot(y_test, 10).astype(np.float32)
    return X_train, y_train_oh, X_test, y_test_oh


def demo_xor():
    """Train on XOR."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    net = NeuralNetwork([2, 8, 1], [ReLU(), Sigmoid()], MSE(), seed=1, output_softmax=False)
    net.fit(X, y, epochs=5000, lr=0.2, batch_size=4, verbose_every=500)
    print("Predictions (prob):", net.predict(X).ravel())
    print("Predicted class:", net.predict_class(X))
    print("True:", y.ravel().astype(int))


def demo_softmax_classification():
    """Train on a 3-class toy dataset."""
    rng = np.random.default_rng(0)
    n = 300
    X0 = rng.normal(loc=(-2, -2), scale=0.6, size=(n, 2))
    X1 = rng.normal(loc=(2, -2), scale=0.6, size=(n, 2))
    X2 = rng.normal(loc=(0, 2), scale=0.6, size=(n, 2))
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n + [1] * n + [2] * n)
    y_oh = one_hot(y, 3).astype(np.float32)

    net = NeuralNetwork([2, 16, 3], [ReLU(), Identity()], CrossEntropy(), seed=2, output_softmax=True)
    net.fit(X, y_oh, epochs=2000, lr=0.05, batch_size=32, verbose_every=200)
    yhat = net.predict_class(X)
    print(f"Training accuracy: {float(np.mean(yhat == y)):.3f}")


def _read_int(prompt: str, default: int) -> int:
    """Read an int from input with a default."""
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid integer, using default={default}.")
        return default


def _read_float(prompt: str, default: float) -> float:
    """Read a float from input with a default."""
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid number, using default={default}.")
        return default


def demo_mnist():
    """Train on MNIST (OpenML) with a simple MLP."""
    print("\nMNIST settings (press Enter to accept defaults):")
    limit_train = _read_int("  Limit train samples [default 20000] : ", 20000)
    limit_test = _read_int("  Limit test samples  [default 10000] : ", 10000)
    epochs = _read_int("  Epochs              [default 5]     : ", 5)
    batch_size = _read_int("  Batch size          [default 64]    : ", 64)
    lr = _read_float("  Learning rate       [default 0.05]  : ", 0.05)
    weight_decay = _read_float("  Weight decay (L2)   [default 1e-4]  : ", 1e-4)

    X_train, y_train, X_test, y_test = load_mnist_openml(
        limit_train=limit_train if limit_train > 0 else None,
        limit_test=limit_test if limit_test > 0 else None,
    )

    net = NeuralNetwork(
        [784, 128, 64, 10],
        [ReLU(), ReLU(), Identity()],
        CrossEntropy(),
        seed=3,
        output_softmax=True,
    )

    for ep in range(1, epochs + 1):
        net.fit(
            X_train,
            y_train,
            epochs=1,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            verbose_every=1,
            shuffle=True,
        )
        train_subset = min(5000, X_train.shape[0])
        train_acc = evaluate(net, X_train[:train_subset], y_train[:train_subset], batch_size=256)
        test_acc = evaluate(net, X_test, y_test, batch_size=256)
        print(f"Epoch {ep:2d} | train_acc~ {train_acc:.3f} | test_acc {test_acc:.3f}")


def main():
    """Menu to run XOR, toy softmax, or MNIST demos."""
    print("Select a demo to run:")
    print("  1) XOR demo")
    print("  2) Softmax classification demo (toy 3-class)")
    print("  3) MNIST training (OpenML via scikit-learn)")
    choice = input("Enter 1/2/3: ").strip()

    if choice == "1":
        demo_xor()
    elif choice == "2":
        demo_softmax_classification()
    elif choice == "3":
        demo_mnist()
    else:
        print("Invalid choice. Please run again and enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
