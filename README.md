# mlax

> [!CAUTION]
> This package is a rather simple and dumb idea, which probably has some sharp edges. Simple programs only for now!

This package supports a single API called `mlax` whose purpose is to transform a _JAX computation_ into one which can be compiled via MLX's `mx.compile` program transformation and executed natively on Apple Metal.

**Example:**
```python
import jax.numpy as jnp
import mlx.core as mx
from mlax import mlax

def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

mx.compile(mlax(jax_code))(mx.array(5.0), mx.array(5.0))
```

The way this API works is that the computation is first staged to a `Jaxpr`, and then an interpreter is run. The interpreter traverses the `Jaxpr`, and replaces JAX primitives (like `jax.lax.add_p`) with ones from [MLX's operation set](https://ml-explore.github.io/mlx/build/html/python/ops.html).

The idea here is that when `mx.compile` performs symbolic tracing through the above process, all the operations which occurs on `mx.array` are directly MLX operations. The result is a transpiler from JAX computations whose primitives have conversion rules (to equivalent MLX operations) to an MLX computation graph.

This can even be used to do fun & dumb things like `mx.compile` a `jax.grad`:
```python
import jax.numpy as jnp
import mlx.core as mx
from mlax import mlax

def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

mx.compile(mlax(jax.grad(jax_code)))(mx.array(5.0), mx.array(5.0))
```
