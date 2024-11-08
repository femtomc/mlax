import functools
from dataclasses import dataclass, field

import beartype.typing as btyping
import jax.core as jc
import jax.numpy as jnp
import jax.tree_util as jtu
import mlx.core as mx
from jax import util as jax_util
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.util import safe_map

from mlax.rules import mlx_rules

Any = btyping.Any
VarOrLiteral = jc.Var | jc.Literal
Callable = btyping.Callable
WrappedFunWithAux = tuple[lu.WrappedFun, Callable[[], Any]]

# We have construct a map between MX types
# and types that JAX can use to specify its arrays dtypes,
# but only for get_shaped_aval below.
dtype_map = {
    mx.int32: int,
    mx.float32: float,
    mx.uint32: jnp.uint32,
}


# Convert MX arrays to a surrogate for JAX tracing.
def get_shaped_aval(x):
    shape, dtype = x.shape, x.dtype
    surrogate = jnp.zeros(shape, dtype_map[dtype])
    return jc.raise_to_shaped(jc.get_aval(surrogate))


# The point of caching here is that, when JAX encounters a function that it needs to convert to a Jaxpr, if it has already done that before, save the work!
@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = jc.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


# The "graph capture" transformation is a "final style" one --
# it has a custom JAX Trace and Tracer type.
# This function is part of that style.
# We don't really use this style in our own transformation, we
# only use one of those transformations (cached_stage_dynamic)
# to get a Jaxpr.
@lu.transformation_with_aux
def _flatten_fun_nokwargs(in_tree, *args_flat):
    py_args = jtu.tree_unflatten(in_tree, args_flat)
    ans = yield py_args, {}
    yield jtu.tree_flatten(ans)


# Wrapper to assign a correct type.
flatten_fun_nokwargs: Callable[[lu.WrappedFun, Any], WrappedFunWithAux] = (
    _flatten_fun_nokwargs  # pyright: ignore[reportAssignmentType]
)


def stage(f):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        typed_jaxpr = cached_stage_dynamic(flat_fun, tuple(flat_avals))
        return typed_jaxpr, (flat_args, in_tree, out_tree)

    return wrapped


###################
# Our interpreter #
###################


@dataclass
class Environment:
    """Keeps track of variables and their values during interpretation."""

    env: dict[int, Any] = field(default_factory=dict)

    def read(self, var: VarOrLiteral) -> Any:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            v = self.env.get(var.count)
            if v is None:
                raise ValueError(
                    f"Unbound variable in interpreter environment at count {var.count}:\nEnvironment keys (count): {list(self.env.keys())}"
                )
            return v

    def get(self, var: VarOrLiteral) -> Any:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Any) -> Any:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.get(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Any:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jc.Literal):
            return True
        return var.count in self.env

    def copy(self):
        keys = list(self.env.keys())
        return Environment({k: self.env[k] for k in keys})


@dataclass
class MLAXInterpreter:
    def _eval_jaxpr_mx(
        self,
        _jaxpr: jc.Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        jax_util.safe_map(env.write, _jaxpr.constvars, consts)
        jax_util.safe_map(env.write, _jaxpr.invars, args)
        for eqn in _jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals

            # Here's where we swap out (what would be) JAX's `eqn.primitive.bind`
            # with our custom rules.
            rule = mlx_rules[eqn.primitive]
            outvals = rule(*args, **params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)

        return jax_util.safe_map(env.read, _jaxpr.outvars)

    def run_interpreter(self, fn, *args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        _closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        _jaxpr, consts = _closed_jaxpr.jaxpr, _closed_jaxpr.literals
        flat_out = self._eval_jaxpr_mx(
            _jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def mlax(f: Callable[..., Any]):
    @functools.wraps(f)
    def wrapped(*args):
        interpreter = MLAXInterpreter()
        return interpreter.run_interpreter(f, *args)

    return wrapped
