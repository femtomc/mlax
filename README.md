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

This can even be used to do fun & dumb things like `mx.compile` a JAX program transformed via `jax.grad`:

```python
import jax.numpy as jnp
import mlx.core as mx
from mlax import mlax

def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

assert jax.grad(composed)(5.0, 5.0) == mx.compile(mlax(jax.grad(composed)))(
    mx.array(5.0),
    mx.array(5.0),
)
```

or even one transformed by `jax.vmap`:

```python
assert all(
    jax.vmap(jax_code)(jnp.ones(5), jnp.ones(5))
    == mx.compile(mlax(jax.vmap(jax_code)))(mx.ones(5), mx.ones(5))
)
```

## State of coverage of JAX primitives

- [X] `lax.add_p`
- [X] `lax.mul_p`
- [X] `lax.sin_p`
- [X] `lax.asinh_p`
- [X] `lax.cos_p`
- [X] `lax.acos_p`
- [X] `lax.abs_p`
- [X] `ad_util.add_any_p`
