from dataclasses import dataclass, field

import beartype.typing as btyping
import jax.core as jc
import mlx.core as mx
from jax import lax
from jax._src import ad_util

Callable = btyping.Callable


@dataclass
class Ruleset:
    mlx_rules: dict[jc.Primitive, Callable[[mx.array, ...], mx.array]] = field(
        default_factory=dict
    )

    def register(self, jax_primitive: jc.Primitive):
        def _register(rule):
            self.mlx_rules[jax_primitive] = rule

        return _register

    def __getitem__(self, jax_primitive: jc.Primitive):
        return self.mlx_rules[jax_primitive]


mlx_rules = Ruleset()

####################
# Registered rules #
####################


@mlx_rules.register(lax.add_p)
def add_mlx(x, y):
    return mx.add(x, y)


@mlx_rules.register(lax.mul_p)
def mul_mlx(x, y):
    return mx.multiply(x, y)


@mlx_rules.register(lax.sin_p)
def sin_mlx(x):
    return mx.sin(x)


@mlx_rules.register(lax.cos_p)
def cos_mlx(x):
    return mx.cos(x)


@mlx_rules.register(ad_util.add_any_p)
def add_any_mlx(x, y):
    return mx.add(x, y)
