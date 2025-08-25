import random

import jax
import numpy as np
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)

num_mcmc = 1
num_walkers = 1600

num_r_up_cart_samples = 5
num_r_dn_cart_samples = 2

r_cart_min, r_cart_max = -1.0, 1.0

r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_walkers, num_r_up_cart_samples, 3) + r_cart_min
r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_walkers, num_r_dn_cart_samples, 3) + r_cart_min

latest_r_up_carts_nw = r_up_carts
latest_r_dn_carts_nw = r_dn_carts
accepted_moves = 0
rejected_moves = 0

# native python
"""
for i_mcmc in range(num_mcmc):
    print(f"i_mcmc={i_mcmc+1}/{num_mcmc}")

    latest_r_up_carts = latest_r_up_carts_nw[0]
    latest_r_dn_carts = latest_r_dn_carts_nw[0]

    # Determine the total number of electrons
    total_electrons = len(latest_r_up_carts) + len(latest_r_dn_carts)

    nbra = 16
    for _ in range(nbra):
        # Choose randomly if the electron comes from up or dn
        if random.randint(0, total_electrons - 1) < len(latest_r_up_carts):
            selected_electron_spin = "up"
            # Randomly select an electron from r_carts_up
            selected_electron_index = random.randint(0, len(latest_r_up_carts) - 1)

            old_r_cart = latest_r_up_carts[selected_electron_index]
        else:
            selected_electron_spin = "dn"
            # Randomly select an electron from r_carts_dn
            selected_electron_index = random.randint(0, len(latest_r_dn_carts) - 1)
            old_r_cart = latest_r_dn_carts[selected_electron_index]

        sigma = 1.0
        g = float(np.random.normal(loc=0, scale=sigma))
        g_vector = np.zeros(3)
        random_index = np.random.randint(0, 3)
        g_vector[random_index] = g
        new_r_cart = old_r_cart + g_vector

        if selected_electron_spin == "up":
            proposed_r_up_carts = latest_r_up_carts.copy()
            proposed_r_dn_carts = latest_r_dn_carts.copy()
            proposed_r_up_carts[selected_electron_index] = new_r_cart
        else:
            proposed_r_up_carts = latest_r_up_carts.copy()
            proposed_r_dn_carts = latest_r_dn_carts.copy()
            proposed_r_dn_carts[selected_electron_index] = new_r_cart

        print(f"  The selected electron is {selected_electron_index+1}-th {selected_electron_spin} electron.")
        print(f"  The selected electron position is {old_r_cart}.")
        print(f"  The proposed electron position is {new_r_cart}.")

        acceptance_ratio = 0.5
        b = np.random.uniform(0, 1)

        if b < acceptance_ratio:
            print("  The proposed move is accepted!")
            accepted_moves += 1
            latest_r_up_carts = proposed_r_up_carts
            latest_r_dn_carts = proposed_r_dn_carts
        else:
            print("  The proposed move is rejected!")
            rejected_moves += 1

    # evaluate observables
    e_L = np.linalg.norm(latest_r_up_carts) + np.linalg.norm(latest_r_dn_carts)
    print(f"  e_L = {e_L}")
"""

key_nw = jnp.array([jax.random.PRNGKey(i * 123) for i in range(num_walkers)])
latest_r_up_carts_nw = jnp.array(latest_r_up_carts_nw)
latest_r_dn_carts_nw = jnp.array(latest_r_dn_carts_nw)

print(key_nw)

# JAX-compatible version
for i_mcmc in range(num_mcmc):
    print(f"i_mcmc={i_mcmc + 1}/{num_mcmc}")

    # @jit
    def run_mcmc(latest_r_up_carts, latest_r_dn_carts, key, A, B):
        # latest_r_up_carts = latest_r_up_carts_nw[0]
        # latest_r_dn_carts = latest_r_dn_carts_nw[0]

        # Determine the total number of electrons
        total_electrons = len(latest_r_up_carts) + len(latest_r_dn_carts)

        nbra = 16
        accepted_moves = 0
        rejected_moves = 0

        for _ in range(nbra):
            # total number of electrons
            total_electrons = len(latest_r_up_carts) + len(latest_r_dn_carts)

            # 0〜total_electrons-1からランダムに選択
            key, subkey = jax.random.split(key)
            rand_num = jax.random.randint(subkey, shape=(), minval=0, maxval=total_electrons)

            # "up"か"dn"を判定するためのブーリアン値
            # is_up == Trueならup、Falseならdn
            is_up = rand_num < len(latest_r_up_carts)

            # upから選ぶ電子index
            key, subkey = jax.random.split(key)
            up_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(latest_r_up_carts))

            # dnから選ぶ電子index
            key, subkey = jax.random.split(key)
            dn_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(latest_r_dn_carts))

            selected_electron_index = jnp.where(is_up, up_index, dn_index)

            # old_r_cartをupかdnから選択
            # 1次元配列なら直接インデックス可能
            old_r_cart = jnp.where(
                is_up, latest_r_up_carts[selected_electron_index], latest_r_dn_carts[selected_electron_index]
            )

            # スピンを数値（0: up, 1: dn）で持つ
            selected_electron_spin = jnp.where(is_up, 0, 1)

            sigma = 1.0

            # 正規分布からサンプル
            key, subkey = jax.random.split(key)
            g = jax.random.normal(subkey, shape=()) * sigma

            # 0〜2の範囲でランダムなインデックスを選択
            key, subkey = jax.random.split(key)
            random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

            # g_vectorにgを代入
            g_vector = jnp.zeros(3)
            g_vector = g_vector.at[random_index].set(g)

            # a dummy heavy calc.
            C = A.dot(B)
            _ = B.dot(B)
            _ = A.dot(C)
            _ = C.dot(C)
            _ = A.dot(A)

            # 位置ベクトルを更新
            new_r_cart = old_r_cart + g_vector

            spin_bool = selected_electron_spin == 1

            proposed_r_up_carts = lax.cond(
                spin_bool,
                lambda _: latest_r_up_carts.at[selected_electron_index].set(new_r_cart),
                lambda _: latest_r_up_carts,
                operand=None,
            )

            proposed_r_dn_carts = lax.cond(
                spin_bool,
                lambda _: latest_r_dn_carts,
                lambda _: latest_r_dn_carts.at[selected_electron_index].set(new_r_cart),
                operand=None,
            )

            print(f"  The selected electron is {selected_electron_index + 1}-th {selected_electron_spin} electron.")
            print(f"  The selected electron position is {old_r_cart}.")
            print(f"  The proposed electron position is {new_r_cart}.")

            acceptance_ratio = 0.5
            key, subkey = jax.random.split(key)
            b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)

            def _accepted_fun(_):
                # Move accepted
                return (accepted_moves + 1, rejected_moves, proposed_r_up_carts, proposed_r_dn_carts)

            def _rejected_fun(_):
                # Move rejected
                return (accepted_moves, rejected_moves + 1, latest_r_up_carts, latest_r_dn_carts)

            # 条件分岐をjax.lax.condで行う
            accepted_moves, rejected_moves, latest_r_up_carts, latest_r_dn_carts = lax.cond(
                b < acceptance_ratio, _accepted_fun, _rejected_fun, operand=None
            )

        return accepted_moves, rejected_moves, latest_r_up_carts, latest_r_dn_carts

    N = 5000
    A = jnp.array(np.random.rand(N, N))
    B = jnp.array(np.random.rand(N, N))

    accepted_moves_nw, rejected_moves_nw, latest_r_up_carts_nw, latest_r_dn_carts_nw = vmap(
        run_mcmc, in_axes=(0, 0, 0, None, None)
    )(latest_r_up_carts_nw, latest_r_dn_carts_nw, key_nw, A, B)
    accepted_moves += jnp.sum(accepted_moves_nw)
    rejected_moves += jnp.sum(rejected_moves_nw)

    print(accepted_moves)
    print(rejected_moves)
    print("-latest_r_up_carts_nw-")
    print(latest_r_up_carts_nw)
    print("-latest_r_dn_carts_nw-")
    print(latest_r_dn_carts_nw)

    ## evaluate observables
    # e_L = np.linalg.norm(latest_r_up_carts) + np.linalg.norm(latest_r_dn_carts)
    # print(f"  e_L = {e_L}")
