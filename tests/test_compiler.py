import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import mlx.core as mx
import pytest

from mlax import mlax


# This is a convenience function designed to support equality
# comparison between executing a function using JAX's JIT
# and executing a function using MX compile.
def jax_equality_assertion(prim_fn, *args):
    dtype_map = {
        mx.int32: int,
        mx.float32: float,
    }

    def convert_to_jax(x):
        return jnp.array(
            x,
            dtype=dtype_map[x.dtype],
        )

    def fn(*args):
        return prim_fn(*args)

    def check(v):
        if not isinstance(v, bool) and v.shape:
            return all(v)
        else:
            return v

    jax_args = jtu.tree_map(convert_to_jax, args)
    assert check(
        pytest.approx(jax.jit(fn)(*jax_args), 1e-5) == mx.compile(mlax(fn))(*args)
    )


def tire_kick_assertion(fn, *args):
    assert mx.any(mx.compile(mlax(fn))(*args))


class TestCompiler:
    def test_add_p(self):
        jax_equality_assertion(lambda x, y: x + y, mx.array(5.0), mx.array(5.0))

    def test_mul_p(self):
        jax_equality_assertion(lambda x, y: x * y, mx.array(5.0), mx.array(5.0))

    def test_sin_p(self):
        jax_equality_assertion(lambda x: jnp.sin(x), mx.array(5.0))

    def test_key_split(self):
        lambda key: jax.random.split(key)
        tire_kick_assertion(lambda key: jax.random.split(key), mx.random.key(1))

    def test_composition(self):
        def composed(x, y):
            v = x + y
            v = v * v
            return jnp.sin(v)

        jax_equality_assertion(composed, mx.array(5.0), mx.array(5.0))

    def test_vmap_composition(self):
        def composed(x, y):
            v = x + y
            v = v * v
            return jnp.sin(v)

        jax_equality_assertion(jax.vmap(composed), mx.ones(5), mx.ones(5))

    def test_grad(self):
        def composed(x, y):
            v = x + y
            v = v * v
            return jnp.sin(v)

        jax_equality_assertion(jax.grad(composed), mx.array(5.0), mx.array(5.0))
