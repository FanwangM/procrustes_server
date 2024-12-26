## About Procrustes Analysis

[Procrustes](https://github.com/theochem/procrustes) is a free, open-source, and cross-platform Python library for (generalized) Procrustes problems with the goal of finding the optimal transformation(s) that makes two matrices as close as possible to each other. This package includes options to translate, scale, and zero-pad matrices, allowing matrices with different centers, scaling, and sizes to be considered.

Please use the following citation in any publication using the Procrustes library:

**"Procrustes: A Python Library to Find Transformations that Maximize the Similarity Between Matrices"**, F. Meng, M. Richer, A. Tehrani, J. La, T. D. Kim, P. W. Ayers, F. Heidar-Zadeh, [Computer Physics Communications, 276(108334), 2022](https://doi.org/10.1016/j.cpc.2022.108334).

### Description of Procrustes Methods

Procrustes problems arise when one wishes to find one or two transformations, $\mathbf{T} \in \mathbb{R}^{n \times n}$ and $\mathbf{S} \in \mathbb{R}^{m \times m}$, that make matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ (input matrix) resemble matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ (target or reference matrix) as closely as possible:

$$
\min_{\mathbf{S}, \mathbf{T}} \| \mathbf{S}\mathbf{A}\mathbf{T} - \mathbf{B} \|_F^2
$$

where the $F$ denotes the Frobenius norm. Here, $a_{ij}$ and $\text{Tr}(\mathbf{A})$ denote the $ij$-th element and trace of matrix $\mathbf{A}$, respectively. When $\mathbf{S}$ is an identity matrix, this is called a **one-sided Procrustes problem**, and when it is equal to $\mathbf{T}$, this becomes a **two-sided Procrustes problem with one transformation**. Otherwise, it is called a **two-sided Procrustes problem**. Different Procrustes problems use different choices for the transformation matrices $\mathbf{S}$ and $\mathbf{T}$, which are commonly taken to be orthogonal/unitary matrices, rotation matrices, symmetric matrices, or permutation matrices. For more detailed information, please refer to the [Procrustes documentation](https://procrustes.qcdevs.org/#).


## Acknowledgments

This webserver is supported by the DRI EDIA Champions Pilot Program and the computational resources of the [Digital Research Alliance](https://alliancecan.ca/).
