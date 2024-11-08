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

        assert mx.compile(mlax(composed))(mx.array(5.0), mx.array(5.0))
