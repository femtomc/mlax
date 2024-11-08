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

    def register(self, jax_primitive: jc.Primitive, mx_primitive):
        self.mlx_rules[jax_primitive] = mx_primitive

    def register_def(self, jax_primitive: jc.Primitive):
        def _register(rule):
            self.mlx_rules[jax_primitive] = rule

        return _register

    def __getitem__(self, jax_primitive: jc.Primitive):
        return self.mlx_rules[jax_primitive]


mlx_rules = Ruleset()

####################
# Registered rules #
####################


mlx_rules.register(lax.add_p, mx.add)
mlx_rules.register(lax.mul_p, mx.multiply)
mlx_rules.register(lax.sin_p, mx.sin)
mlx_rules.register(lax.cos_p, mx.cos)
mlx_rules.register(lax.abs_p, mx.abs)
mlx_rules.register(lax.all_p, mx.all)
mlx_rules.register(ad_util.add_any_p, mx.add)
