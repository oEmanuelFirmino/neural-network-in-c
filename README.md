# Neural Network from Scratch in C (XOR Problem)

![Language](https://img.shields.io/badge/Language-C-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)

This project is a comprehensive, academic-level implementation of a feedforward neural network, written entirely in **pure C** with no external machine learning libraries. Its primary goal is to serve as an educational tool, clearly demonstrating the foundational concepts of neural networks from first principles. By building everything from scratch—from memory management to the learning algorithm—this repository provides a deep, practical understanding of **forward propagation**, **backpropagation**, and **gradient descent**. The network is trained to solve the classic XOR problem, a non-linearly separable task that perfectly illustrates the power of multi-layer perceptrons.

## ✨ Core Features

- **Zero Dependencies:** Built exclusively with the standard C library (`stdio`, `stdlib`, `math.h`, `time.h`), ensuring maximum portability and focus on the core algorithms.
- **Flexible, Dynamic Architecture:** The network's topology (number of layers and neurons per layer) is defined at runtime, allowing for easy experimentation with different architectures without recompiling the code.
- **Multiple Activation Functions:** Includes implementations for **Sigmoid** and **ReLU (Rectified Linear Unit)**, along with their exact analytical derivatives required for precise gradient calculations during backpropagation.
- **Backpropagation from Scratch:** The complete learning algorithm is implemented manually. This includes the calculation of error signals (deltas) and the application of the chain rule to propagate these signals backward through the network, providing deep insight into how a network truly learns.
- **Robust Memory Management:** A pair of dedicated functions (`create_network` and `release_network`) handles all `malloc` and `free` calls, ensuring a clean and predictable memory footprint and preventing common issues like memory leaks.

---

## 🚀 Getting Started

To compile and run this project, you will need a standard C compiler (like GCC or Clang) installed on your system. The code is platform-independent and should work on any major operating system.

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/oEmanuelFirmino/neural-network-in-c.git](https://github.com/oEmanuelFirmino/neural-network-in-c.git)
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd neural-network-in-c
    ```

3.  **Compile the code:**
    Use the following command to compile the `main.c` source file. The `-lm` flag is crucial for linking the math library, which provides the `exp()` and `pow()` functions used in the Sigmoid activation and error calculation, respectively.

    ```bash
    gcc main.c -o neural_network -lm
    ```

4.  **Run the executable:**
    ```bash
    ./neural_network
    ```

### ➡ Expected Output

When you run the program, you will observe the training process in real-time, with the average error reported every 1000 epochs. This allows you to monitor the network's convergence. Once training is complete, the final section will show the network's predictions for each XOR input, demonstrating its learned capability.

```
Starting training...
Epoch 1000/10000, Error=0.252084
Epoch 2000/10000, Error=0.248661
Epoch 3000/10000, Error=0.187219
...
Epoch 10000/10000, Error=0.000392
Training complete.

Testing the trained network:
Input: [0.0, 0.0] -> Prediction: 0.009384 (Expected: 0.0)
Input: [0.0, 1.0] -> Prediction: 0.982276 (Expected: 1.0)
Input: [1.0, 0.0] -> Prediction: 0.982361 (Expected: 1.0)
Input: [1.0, 1.0] -> Prediction: 0.023277 (Expected: 0.0)

Resources freed.
```

---

## 🛠️ Code Architecture Explained

The code is structured modularly to promote clarity and maintainability. It separates the network's static definition (data structures), its lifecycle management (creation/deletion), and its dynamic operations (propagation/training).

### 1. Data Structures (`structs`)

- **`Neuron`**: The fundamental computational unit of the network. It encapsulates all state related to a single neuron:
  - `weights`: A dynamic array of `double`, holding the connection weights from the neurons in the previous layer.
  - `bias`: A single `double` value that shifts the activation function.
  - `output`: The value computed during the forward pass after applying the activation function.
  - `delta`: The calculated error term for this neuron during backpropagation. This value is critical for calculating the gradients.
- **`Layer`**: A logical grouping of neurons. It is essentially a container holding a dynamic array of `Neuron` structs and a count of how many neurons it contains.
- **`NeuralNetwork`**: The master struct that represents the entire network. It holds an array of `Layer` structs and, importantly, function pointers for the `activation_function` and its `derivative_activation`. This design choice makes it trivial to switch out activation functions for the entire network.

### 2. Network Management

- **`create_network()`**: This function acts as the network's constructor. It orchestrates the full memory allocation process based on a given topology array (e.g., `{2, 2, 1}` for an XOR network). It iterates through layers and neurons, allocating memory and initializing all weights with small random values. Random initialization is a critical step to break symmetry; if all weights were initialized to the same value, all neurons in a layer would learn the same features. Biases are initialized to zero.
- **`release_network()`**: This is the network's destructor. It meticulously frees all dynamically allocated memory in the reverse order of creation (neuron weights, then neurons, then layers, then the network itself) to prevent any memory leaks.

### 3. Forward Propagation

This is the inference phase where the network generates an output from a given input. The process flows sequentially from the input layer to the output layer.

1.  For each neuron in a layer (starting from the first hidden layer), we calculate its weighted sum, also known as the **net input** or **logit** ($z$). This is the dot product of the input vector (which is the output vector of the previous layer) and the neuron's weight vector, plus the neuron's bias.
    $z = \left( \sum_{i=1}^{n} w_i \cdot \text{input}_i \right) + b$
2.  The net input $z$ is then passed through a non-linear **activation function** ($\sigma$) to produce the neuron's final output ($a$). This non-linearity is what allows the network to learn complex relationships in data that linear models cannot.
    $a = \sigma(z)$
3.  The vector of outputs from one layer becomes the input vector for the subsequent layer. This process is repeated until the final layer produces the network's ultimate prediction.

### 4. Backpropagation and Training

This is the core of the learning process, where the network iteratively adjusts its parameters (weights and biases) to minimize its prediction error. It consists of a backward pass followed by a parameter update.

1.  **Calculate Total Error**: After a forward pass, we quantify how "wrong" the network's prediction was using a **cost function**. The code uses the **Sum of Squared Errors (SSE)** for a single sample, which is a common choice for regression-style problems.
    $E = \frac{1}{2} \sum_{k} (\text{expected}_k - \text{prediction}_k)^2$
    The factor of $\frac{1}{2}$ is a mathematical convenience that cancels out during differentiation, simplifying the gradient calculation.
2.  **Calculate the Output Layer Delta (`calculate_delta_output`)**: The backward pass begins here. We determine how much each output neuron contributed to the total error by calculating the **error term** (delta, $\delta$) for each neuron in the output layer. This term represents the gradient of the cost function with respect to the neuron's net input $z$.
    $\delta_{\text{output}} = (\text{prediction} - \text{expected}) \odot \sigma'(\text{output})$
    Here, $(\text{prediction} - \text{expected})$ is the derivative of the error with respect to the neuron's output, and $\sigma'(\text{output})$ is the derivative of the activation function. The $\odot$ symbol denotes element-wise multiplication.
3.  **Propagate the Error Backwards (`propagate_error_backwards`)**: We then recursively calculate the `delta` for each hidden layer, moving from the last hidden layer towards the first. The error of a hidden neuron is the **sum of the errors of the next layer's neurons, weighted by the strength of their connections**. This step is a direct application of the chain rule.
    $\delta_{\text{hidden}} = \left( \sum_{k} w_{jk} \cdot \delta_k \right) \odot \sigma'(\text{output}_{\text{hidden}})$
    This process effectively distributes the responsibility for the total error back through the network's connections.
4.  **Update Parameters (`update_parameters`)**: With the deltas for every neuron calculated, we have the necessary information to update the weights and biases using **gradient descent**. The update rule moves each parameter a small step in the opposite direction of its gradient.
    - **Weight Update**: $w_{ij} \leftarrow w_{ij} - \eta \cdot \delta_j \cdot \text{input}_i$
    - **Bias Update**: $b_j \leftarrow b_j - \eta \cdot \delta_j$
      Where $\eta$ is the **learning rate**, a critical hyperparameter that controls the step size. A small learning rate leads to slow but stable convergence, while a large one can cause the training to overshoot the minimum and diverge.

---

## 🧠 The Mathematics of Backpropagation

Backpropagation is not a new algorithm; it is a clever application of the **chain rule** from multivariable calculus, optimized for neural networks. It allows for the efficient computation of the gradient of a complex, nested function (the network) with respect to its parameters.

#### 1. Gradient for Output Layer Weights

To update a weight $w_{jk}$ connecting a hidden neuron $j$ to an output neuron $k$, we need the partial derivative of the error $E$ with respect to $w_{jk}$. The chain rule breaks this down into a product of simpler derivatives:

$\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial a_k} \cdot \frac{\partial a_k}{\partial z_k} \cdot \frac{\partial z_k}{\partial w_{jk}}$

Let's inspect each term:

- $\frac{\partial E}{\partial a_k} = (a_k - y_k)$: This term tells us how the error changes with respect to the activated output $a_k$. ($y_k$ is the target value).
- $\frac{\partial a_k}{\partial z_k} = \sigma'(z_k)$: This term tells us how the activation responds to changes in the net input $z_k$.
- $\frac{\partial z_k}{\partial w_{jk}} = a_j$: This term tells us how the net input $z_k$ is affected by the weight $w_{jk}$. Since $z_k = \sum_i w_{ik} a_i + b_k$, the derivative is simply the input to that weight, which is the output $a_j$ of the neuron from the previous layer.

Combining these, we get: $\frac{\partial E}{\partial w_{jk}} = (a_k - y_k) \cdot \sigma'(z_k) \cdot a_j$.
To simplify, we group the first two terms into the **error delta**: $\delta_k = (a_k - y_k) \cdot \sigma'(z_k)$. This simplifies the final gradient expression to:

$\frac{\partial E}{\partial w_{jk}} = \delta_k \cdot a_j$

This elegant result is the value used in the gradient descent update rule: `new_weight = old_weight - learning_rate * gradient`.

#### 2. Gradient for Hidden Layer Weights

For a weight $w_{ij}$ in a hidden layer, the logic is similar but involves an additional summation, because this weight affects the final error through _all_ the neurons in the next layer that it connects to.

$\frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial a_j} \cdot \frac{\partial a_j}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}$

The new, complex term is $\frac{\partial E}{\partial a_j}$. The output $a_j$ influences every neuron $k$ in the next layer. Therefore, we must sum its influence on the total error through all these paths:
$\frac{\partial E}{\partial a_j} = \sum_k \frac{\partial E}{\partial z_k} \cdot \frac{\partial z_k}{\partial a_j} = \sum_k \delta_k \cdot w_{jk}$

This gives us the formula for the hidden layer delta:
$\delta_j = \left( \sum_k \delta_k w_{jk} \right) \sigma'(z_j)$

This is the mathematical essence of "backpropagation": the hidden layer's error signal ($\delta_j$) is the weighted sum of the subsequent layer's error signals ($\delta_k$). Once $\delta_j$ is known, the gradient calculation for $w_{ij}$ follows the same simple form as before: $\frac{\partial E}{\partial w_{ij}} = \delta_j \cdot a_i$.

---

## 🔮 Future Improvements

While this project provides a solid and complete foundation, it could be extended in several exciting directions to explore more advanced concepts in deep learning:

- **Implement Advanced Optimizers:** Go beyond standard gradient descent by implementing optimizers like **Momentum**, which helps accelerate convergence, or **Adam**, an adaptive learning rate method that is the de facto standard in modern deep learning.
- **Support for Batch and Mini-Batch Training:** Modify the training loop to support mini-batch gradient descent. This offers a balance between the accuracy of batch gradient descent and the speed of stochastic gradient descent, and it is the most common training method used today.
- **Add More Cost Functions:** Implement additional cost functions, such as **Cross-Entropy**, which is mathematically better suited for classification tasks and often leads to faster training than SSE.
- **Modularize and Load Data:** Create functions to load training data, labels, and network topologies from external files (e.g., CSV, JSON). This would decouple the model from the data and make it a more versatile and reusable tool.
- **Introduce a Makefile:** Add a `Makefile` to automate the compilation process, manage dependencies, and provide clean build/rebuild commands, which is standard practice for larger C projects.
- **Regularization:** Implement techniques like L1 or L2 regularization to prevent overfitting by adding a penalty term for large weights to the cost function.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
