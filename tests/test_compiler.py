import jax
import jax.numpy as jnp
import mlx.core as mx

from mlax import mlax


class TestCompiler:
    def test_add(self):
        assert 10 == mx.compile(mlax(lambda x, y: x + y))(
            mx.array(5),
            mx.array(5),
        )

    def test_mul(self):
        assert 25 == mx.compile(mlax(lambda x, y: x * y))(
            mx.array(5),
            mx.array(5),
        )

    def test_composition(self):
        def composed(x, y):
            v = x + y
            v = v * v
            return jnp.sin(v)

        assert composed(5.0, 5.0) == mx.compile(mlax(composed))(
            mx.array(5.0),
            mx.array(5.0),
        )

    def test_vmap_composition(self):
        def composed(x, y):
            v = x + y
            v = v * v
            return jnp.sin(v)

        assert all(
            jax.vmap(composed)(jnp.ones(5), jnp.ones(5))
            == mx.compile(mlax(jax.vmap(composed)))(mx.ones(5), mx.ones(5))
        )

    def test_grad(self):
        def composed(x, y):
            v = x + y
            v = v * v
            return jnp.sin(v)

        assert jax.grad(composed)(5.0, 5.0) == mx.compile(mlax(jax.grad(composed)))(
            mx.array(5.0),
            mx.array(5.0),
        )
