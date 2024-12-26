## About Procrustes Analysis

[Procrustes](https://github.com/theochem/procrustes) is a free, open-source, and cross-platform Python library for (generalized) Procrustes problems with the goal of finding the optimal transformation(s) that makes two matrices as close as possible to each other. This package includes options to translate, scale, and zero-pad matrices, allowing matrices with different centers, scaling, and sizes to be considered.

Please use the following citation in any publication using the Procrustes library:

**"Procrustes: A Python Library to Find Transformations that Maximize the Similarity Between Matrices"**, F. Meng, M. Richer, A. Tehrani, J. La, T. D. Kim, P. W. Ayers, F. Heidar-Zadeh, [Computer Physics Communications, 276(108334), 2022](https://doi.org/10.1016/j.cpc.2022.108334).

### Description of Procrustes Methods

Procrustes problems arise when one wishes to find one or two transformations, $\mathbf{T} \in \mathbb{R}^{n \times n}$ and $\mathbf{S} \in \mathbb{R}^{m \times m}$, that make matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ (input matrix) resemble matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ (target or reference matrix) as closely as possible:

$$
\min_{\mathbf{S}, \mathbf{T}} \| \mathbf{S}\mathbf{A}\mathbf{T} - \mathbf{B} \|_F^2
$$

where the $F$ denotes the Frobenius norm. Here, $a_{ij}$ and $\text{Tr}(\mathbf{A})$ denote the $ij$-th element and trace of matrix $\mathbf{A}$, respectively. When $\mathbf{S}$ is an identity matrix, this is called a **one-sided Procrustes problem**, and when it is equal to $\mathbf{T}$, this becomes a **two-sided Procrustes problem with one transformation**. Otherwise, it is called a **two-sided Procrustes problem**. Different Procrustes problems use different choices for the transformation matrices $\mathbf{S}$ and $\mathbf{T}$, which are commonly taken to be orthogonal/unitary matrices, rotation matrices, symmetric matrices, or permutation matrices.


| **Procrustes Type**                      | $\mathbf{S}$                 | $\mathbf{T}$               | **Constraints**                                                                                                                                             |
|------------------------------------------|--------------------------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Generic                                  | $\mathbf{I}$                 | $$\mathbf{T}$$               | None                                                                                                                                                        |
| Orthogonal                               | $\mathbf{I}$                 | $$\mathbf{Q}$$               | $\mathbf{Q}^{-1} = \mathbf{Q}^\dagger$                                                                                                                      |
| Rotational                               | $\mathbf{I}$                 | $$\mathbf{R}$$               | $\begin{cases} \mathbf{R}^{-1} = \mathbf{R}^\dagger \\\\ \mathbf{R}^\top \mathbf{R} = \mathbf{I} \end{cases}$                                               |
| Symmetric                                | $\mathbf{I}$                 | $$\mathbf{X}$$               | $\mathbf{X} = \mathbf{X}^\dagger$                                                                                                                           |
| Permutation                              | $\mathbf{I}$                 | $$\mathbf{P}$$               | $$\begin{cases} {\mathbf{P}}_{ij} \in \left\{0, 1\right\} \\\\ \sum_{i=1}^n {\mathbf{P}}_{ij} = \sum_{j=1}^n {\mathbf{P}}_{ij} = 1 \end{cases}$$                          |
| Two-sided Orthogonal                     | $\mathbf{Q}_1^\dagger$      | $$\mathbf{Q}_2$$             | $$\begin{cases} \mathbf{Q}_1^{-1} = \mathbf{Q}_1^\dagger \\\\ \mathbf{Q}_2^{-1} = \mathbf{Q}_2^\dagger \end{cases} $$                                          |
| Two-sided Orthogonal with One Transform | $\mathbf{Q}^\dagger$        | $$\mathbf{Q}$$               | $\mathbf{Q}^{-1} = \mathbf{Q}^\dagger$                                                                                                                      |
| Two-sided Permutation                    | $\mathbf{P}_1^\dagger$      | $$\mathbf{P}_2$$             | $$\begin{cases} {\mathbf{P}_1}_{ij} \in \{0, 1\} \\\\ {\mathbf{P}_2}_{ij} \in \{0, 1\} \\\\ \sum_{i=1}^n {\mathbf{P}_1}_{ij} = \sum_{j=1}^n {\mathbf{P}_1}_{ij} = 1 \\\\ \sum_{i=1}^n {\mathbf{P}_2}_{ij} = \sum_{j=1}^n {\mathbf{P}_2}_{ij} = 1 \end{cases}$$ |
| Two-sided Permutation with One Transform| $\mathbf{P}^\dagger$        | $$\mathbf{P}$$               | $$\begin{cases} \mathbf{P}_{ij} \in \{0, 1\} \\\\ \sum_{i=1}^n \mathbf{P}_{ij} = \sum_{j=1}^n \mathbf{P}_{ij} = 1 \end{cases}$$                          |


In addition to these Procrustes methods, the [generalized Procrustes analysis (GPA)](#generalized-procrustes-analysis) and the softassign algorithm are also implemented in our package.

#### Generalized Procrustes Analysis (GPA)

The GPA algorithm seeks the optimal transformation matrices $\mathbf{T}$ to superpose the given objects (usually more than 2) with minimum distance:

$$
\min \sum_{i<j} \| \mathbf{A}_i \mathbf{T}_i - \mathbf{A}_j \mathbf{T}_j \|_F^2
$$

where $\mathbf{A}_i$ and $\mathbf{A}_j$ are the configurations and $\mathbf{T}_i$ and $\mathbf{T}_j$ denote the transformation matrices for $\mathbf{A}_i$ and $\mathbf{A}_j$, respectively. When only two objects are given, the problem reduces to a generic Procrustes problem.

#### Softassign Algorithm

The softassign algorithm was first proposed to deal with the quadratic assignment problem. The objective function minimizes $E_{qap}(\mathbf{M}, \mu, \nu)$, which is defined as:

$$
E_{qap}(\mathbf{M}, \mu, \nu) = -\frac{1}{2} \sum_{ai,bj} \mathbf{C}_{ai; bj} \mathbf{M}_{ai} \mathbf{M}_{bj} + \sum_a \mu_a \left(\sum_i \mathbf{M}_{ai} - 1 \right) + \sum_i \nu_i \left(\sum_a \mathbf{M}_{ai} - 1 \right) - \frac{\gamma}{2} \sum_{ai} \mathbf{M}_{ai}^2 + \frac{1}{\beta} \sum_{ai} \mathbf{M}_{ai} \log \mathbf{M}_{ai}
$$

This algorithm can handle the two-sided permutation Procrustes problem, a special case of the quadratic assignment problem.


## Acknowledgments

This webserver is supported by the DRI EDIA Champions Pilot Program and the computational resources of the [Digital Research Alliance](https://alliancecan.ca/).
