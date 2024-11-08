import abc
import functools
import itertools as it
from dataclasses import dataclass, field

import beartype.typing as btyping
import jax.core as jc
import jax.tree_util as jtu
import mlx
from jax import tree_util
from jax import util as jax_util
from jax.extend import linear_util as lu
from jax.interpreters import batching, mlir
from jax.interpreters import partial_eval as pe
from jax.util import safe_map

Any = btyping.Any
VarOrLiteral = jc.Var | jc.Literal
Callable = btyping.Callable
WrappedFunWithAux = tuple[lu.WrappedFun, Callable[[], Any]]


def get_shaped_aval(x):
    return jc.raise_to_shaped(jc.get_aval(x))


@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = jc.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


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


@dataclass
class Environment:
    """Keeps track of variables and their values during propagation."""

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


mlx_rules: dict[jc.Primitive, Callable[[mlx.Array, ...], mlx.Array]] = {}


@dataclass
class MLAXInterpreter:
    def _eval_jaxpr_forward(
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
        flat_out = self._eval_jaxpr_forward(
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
