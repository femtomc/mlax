# mlax

> [!CAUTION]
> This package is a rather simple and dumb idea, which probably has some sharp edges.

This package supports a single API called `mlax` whose purpose is to transform a _JAX computation_ into one which can be compiled via MLX's `mx.compile` program transformation and executed natively on Apple Metal.

Example:
```python
import jax.numpy as jnp
import mlx.core as mx
from mlax import mlax

def composed(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

mx.compile(mlax(composed))(mx.array(5.0), mx.array(5.0))
```
