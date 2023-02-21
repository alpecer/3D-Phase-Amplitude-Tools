# Global Phase Amplitude Description of Oscillatory Dynamics via the Parameterization Method

The code in this repository is based on the results published in "Global phase-amplitude description of oscillatory dynamics via the parameterization method" [[Published Version]](https://aip.scitation.org/doi/abs/10.1063/5.0010149) [[arXiv]](https://arxiv.org/pdf/2004.03647.pdf)

## Theoretical background

Having the following system
$$\dot{x} = X(x), \qquad x \in \mathbb{R}^{d}, \qquad d \geq 2 ,$$
with an underlying limit cycle, we look for the change of variables $x=K(\theta, \sigma)$ such that the dynamics express simply as
```math
\dot{\theta} = \frac{1}{T}, \quad \quad \quad  \dot{\sigma} = \Lambda \cdot \sigma, \quad \quad \text{with} \quad \quad \Lambda = 
\begin{pmatrix}
\lambda_1 & & \\
& \ddots & \\
& & \lambda_{d-1}  
\end{pmatrix}
```
with $T$ the period of the limit cycle and $\lambda_i$ its correspondent Floquet exponents.

To find such a parameterisation $x=K(\theta, \sigma)$ one has to solve the following invariance equation
```math
\frac{1}{T}\frac{\partial}{\partial\theta}K(\theta, \sigma) + \sum_{i=1}^{d-1} \lambda_i \sigma_i \frac{\partial}{\partial \sigma_i}K(\theta, \sigma) = X(K(\theta, \sigma))
```

The code we next present, solves the above mentioned invariance equation for the case $d=3$ and provides an expression for the parameterisation $x=K(\theta, \sigma)$ in the following form
$$K(\theta, \sigma) = \sum_{m=0}^{\infty} \sum_{\alpha=0}^{m} K_{\alpha, m-\alpha}(\theta) \sigma^{\alpha}_1 \sigma^{m-\alpha}_2$$

Our code is efficient as it relies on
- a diagonalisation of the linear system yielding the coeficcients $K_{\alpha, m-\alpha}(\theta)$ by means of the Floquet normal form
- automatic differentiation techniques.

## How to use it?

Our code considers a 3D neuron model (see Eq. 59) but it is easy to adapt for any system of interest. To that end, one just needs to
- Define its vectror field equations at 
- Define the jacobian matrix at
- introduce the period
- introduce an initial condition at the limit cycle
- write the automatic differentiation X for your system

There are extra parameters $b_1, b_2$ controlling the norm of $K_{10}$ and $K_{01}$ (see Remark 3.2 in the manuscript for more details)

## What produces?

By running the code you will generate three files X each of them containing its corresponding
