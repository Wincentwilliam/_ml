# Backpropagation and Computational Graphs

## 1. Overview
This project demonstrates the application of **Backpropagation**—a foundational algorithm in neural networks—to calculate gradients for two distinct mathematical functions. By constructing a **Computational Graph**, we can decompose complex operations into simple, differentiable steps.

## 2. Theoretical Framework

### Computational Graph
A computational graph maps the flow of data through mathematical operations. Each node represents an operation (e.g., addition, multiplication) or an input variable, while the edges represent the flow of tensors.

### Chain Rule of Calculus
The backpropagation algorithm relies on the **Chain Rule**, which allows us to calculate the gradient of a composite function. For a function $f = g(h(x))$, the derivative is:
$$\frac{df}{dx} = \frac{df}{dh} \cdot \frac{dh}{dx}$$
In our graph, we propagate the "upstream" gradient from the output back to the input nodes.

---

## 3. Mathematical Analysis

### Function 1: $f(x, y, z) = (x \cdot y) + z$
Given inputs: $x=2, y=3, z=4$.

| Node | Forward Value (val) | Gradient (grad) |
| :--- | :--- | :--- |
| **x** | 2 | 3 |
| **y** | 3 | 2 |
| **z** | 4 | 1 |
| **(x * y)** | 6 | 1 |
| **f** | 10 | 1 |

*   **Gradient Formula:** $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial (*)} \cdot \frac{\partial (*)}{\partial x} = 1 \cdot y = 3$

### Function 2: $f(x, y, z, t) = ((x \cdot y) + z) \cdot t$
Given inputs: $x=2, y=3, z=4, t=5$.

| Node | Forward Value (val) | Gradient (grad) |
| :--- | :--- | :--- |
| **x** | 2 | 15 |
| **y** | 3 | 10 |
| **z** | 4 | 5 |
| **t** | 5 | 10 |
| **(x * y)** | 6 | 5 |
| **((x * y) + z)** | 10 | 5 |
| **f** | 50 | 1 |

*   **Gradient Formula:** $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial (*_2)} \cdot \frac{\partial (*_2)}{\partial (+)} \cdot \frac{\partial (+)}{\partial (*_1)} \cdot \frac{\partial (*_1)}{\partial x} = 1 \cdot 5 \cdot 1 \cdot 3 = 15$

---

## 4. Conclusion
The backpropagation process effectively computes the sensitivity of the final output $f$ with respect to each input parameter. By utilizing the computational graph, we ensure modularity and computational efficiency, which is the core principle behind modern machine learning frameworks like PyTorch and TensorFlow.