import jax.numpy as jnp
import numpy as np
from jax import grad, jacrev
from scipy.sparse import bmat, csr_matrix


# numpy and scipy implementation
def compute_sparse_mul(A, x):
    # A, sparse matrix
    # x, vector
    return A.dot(x)


def compute_sparse_mul_jnp(A, x):
    # A, sparse_matrix_list
    # x, vector
    return jnp.dot(A, x)


# 例として、3つのブロック正方行列を持つ場合
num_block = 3
size_block_list = [2, 3, 4]

# 空のリストを用意して、疎行列のブロックを格納
blocks = []

for i in range(num_block):
    row_blocks = []
    for j in range(num_block):
        if i == j:
            # 対角部分はランダムな正方行列を疎行列として生成
            block = csr_matrix(np.random.rand(size_block_list[i], size_block_list[i]))
        else:
            # 対角以外の部分はすべてゼロの疎行列
            block = csr_matrix((size_block_list[i], size_block_list[j]))
        row_blocks.append(block)
    blocks.append(row_blocks)

# scipy.sparse.bmatを使って疎ブロック行列を作成
sparse_block_matrix = bmat(blocks, format="csr")

# ベクトルを作成（疎行列の列数に合わせたベクトルを生成）
vector = np.random.rand(sparse_block_matrix.shape[1])

print(sparse_block_matrix)

print(compute_sparse_mul(sparse_block_matrix, vector))

# JAX implementation (蜜行列で扱うしかない...)

# ブロック正方行列を格納するリスト
blocks = []

# 指定されたサイズでランダムな正方行列を生成
for size in size_block_list:
    block = np.random.rand(size, size)
    blocks.append(jnp.array(block))

print(blocks)

"""
A = jnp.array([[1, 2, 3], [2, 3, 4]])
B = jnp.array([[1, 2], [3, 4]])
C = jnp.block([A, B])
print(C)
"""
