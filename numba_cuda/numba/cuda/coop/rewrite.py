# cuda.cooperative

from numba.core.typing.templates import CallableTemplate

from enum import IntEnum
from functools import cached_property
from collections import OrderedDict, defaultdict
import inspect
from numba.core.typing.templates import (
    Signature,
    make_callable_template,
    make_concrete_template,
)

# coop_rewrite.py  –  put this on your PYTHONPATH
from numba.core import ir, types, ir_utils
from numba.core.typing import signature
from numba.core.rewrites import register_rewrite, Rewrite
from numba import cuda
from dataclasses import dataclass

from numba.core import config

from . import block

config.DEBUG = True
config.DUMP_IR = True
config.CUDA_ENABLE_PYNVJITLINK = True

class Granularity(IntEnum):
    """
    Enum for the granularity of the cooperative operation.
    """

    BLOCK = 0
    WARP = 1
    THREAD = 2


class Primitive(IntEnum):
    """
    Enum for the primitive type of the cooperative operation.
    """

    LOAD = 0
    STORE = 1
    SCAN = 2
    REDUCE = 3


@dataclass
class CoopStmt:
    index: int
    block_line: int
    expr: ir.Expr
    instr: ir.Assign
    template: types.Any
    func: types.Any
    func_name: str
    impl_class: types.ClassType
    target_name: str
    calltype: types.Any

    # Defaults.
    implicit_temp_storage: bool = True

    # Provided after the fact by the rewrite pass.
    dtype: types.DType = None
    dim: types.Any = None
    items_per_thread: int = None
    algorithm_id: int = None
    runtime_args: tuple = None
    expr_args: list = None
    expr_args_no_longer_needed: list = None
    src: types.Any = None
    dst: types.Any = None
    invocable: types.Any = None
    instance: types.Any = None

    @cached_property
    def call_var_name(self):
        return (
            f"{self.granularity.name.lower()}_"
            f"{self.primitive.name.lower()}_"
            f"{self.block_line}_{self.index}"
        )

    @cached_property
    def expr_name(self):
        return f"{self.granularity.name.lower()}_{self.primitive.name.lower()}"

    @cached_property
    def granularity(self):
        """
        Determine the granularity of the cooperative operation.
        """
        template_name = self.template.__name__
        if "block" in template_name:
            return Granularity.BLOCK
        elif "warp" in template_name:
            return Granularity.WARP
        else:
            raise RuntimeError(f"Unknown granularity: {self!r}")

    @cached_property
    def primitive(self):
        """
        Determine the primitive type of the cooperative operation.
        """
        template_name = self.template.__name__
        if "load" in template_name:
            return Primitive.LOAD
        elif "store" in template_name:
            return Primitive.STORE
        elif "scan" in template_name:
            return Primitive.SCAN
        elif "reduce" in template_name:
            return Primitive.REDUCE
        else:
            raise RuntimeError(f"Unknown primitive: {self!r}")

    @property
    def is_load(self):
        return self.primitive == Primitive.LOAD

    @property
    def is_store(self):
        return self.primitive == Primitive.STORE

    @property
    def is_load_or_store(self):
        return (
            self.primitive == Primitive.LOAD
            or self.primitive == Primitive.STORE
        )

    @property
    def is_block(self):
        return self.granularity == Granularity.BLOCK

    @property
    def is_warp(self):
        return self.granularity == Granularity.WARP


def _lower_block_load_or_store(lowerer, expr):
    """
    Lower a block load or store expression to the appropriate
    cooperative implementation.
    """
    context = lowerer.context
    builder = lowerer.builder

    cs = expr.coop_stmt
    algo = cs.instance.specialization
    print(f"Lowering {cs.template} with specialization: {algo}")



@register_rewrite("after-inference")
class InterceptCooperativeCalls(Rewrite):
    """
    Stage-5a pass that intercepts every call to any
    cuda.<block|warp>.<load|store> intrinsic.

    * `match()` is run once per basic-block – keep it cheap.
    * `apply()` is only entered if `match()` returned True.
    """

    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)

        from ..cudadecl import (
            Coop_block_load,
            Coop_block_store,
            # Coop_warp_load,
            # Coop_warp_store,
        )

        self._coop_templates = {
            Coop_block_load: block.load,
            Coop_block_store: block.store,
            # Coop_warp_load: warp.load,
            # Coop_warp_store: warp.store,
        }

        special_ops = state.targetctx.special_ops
        if "block_load" not in special_ops:
            special_ops["block_load"] = _lower_block_load_or_store
        if "block_store" not in special_ops:
            special_ops["block_store"] = _lower_block_load_or_store

        self.remove_map = {}
        self.match_count = 0
        self.apply_count = 0

    def match(self, func_ir, block, typemap, calltypes, **kw):
        # If there are no calls in this block, we can immediately skip it.
        num_calltypes = len(calltypes)
        if num_calltypes == 0:
            return False

        self.match_count += 1

        first = True
        found = False

        # assignments = block.find_insts(ir.Assign)
        for i, instr in enumerate(block.body):
            if not isinstance(instr, ir.Assign):
                continue
            expr = instr.value
            if not isinstance(expr, ir.Expr):
                continue
            if expr.op != "call":
                continue
            func_name = expr.func.name
            func = typemap[func_name]
            templates = func.templates
            # XXX: are there any circumstances where there are multiple
            # templates?
            if len(templates) > 1:
                assert len(templates) == 2, (len(templates), templates)
                template = templates[1]
            else:
                template = templates[0]

            impl_class = self._coop_templates.get(template, None)
            if not impl_class:
                continue

            if first:
                self.func_ir = func_ir
                self.block = block
                self.typemap = typemap
                self.calltypes = calltypes
                self.coop_assigns = OrderedDict()
                first = False

            target_name = instr.target.name

            coop_stmt = CoopStmt(
                index=i,
                block_line=block.loc.line,
                expr=expr,
                instr=instr,
                template=template,
                func=func,
                func_name=func_name,
                impl_class=impl_class,
                target_name=target_name,
                calltype=calltypes[expr],
            )

            assert target_name not in self.coop_assigns, target_name
            self.coop_assigns[target_name] = coop_stmt
            self._refine_match(coop_stmt)
            found = True

        return found

    def _refine_match(self, coop_stmt):
        if coop_stmt.is_load_or_store:
            self._refine_match_load_or_store(coop_stmt)

    def _refine_match_load_or_store(self, coop_stmt):
        """
        Handle a load or store instruction by replacing it with a call to the
        cooperative implementation.
        """
        cs = coop_stmt
        expr = cs.expr

        dtype = None
        dim = None
        items_per_thread = None
        algorithm = None
        algorithm_id = None
        expr_args = cs.expr_args = list(expr.args)
        expr_args_no_longer_needed = cs.expr_args_no_longer_needed = []
        if cs.is_load:
            src = expr_args.pop(0)
            dst = expr_args.pop(0)
            runtime_args = [src, dst]
        else:
            dst = expr_args.pop(0)
            src = expr_args.pop(0)
            runtime_args = [dst, src]

        arg_ty = self.typemap[src.name]
        assert isinstance(arg_ty, types.Array)
        dtype = arg_ty.dtype

        # Dim should be next.
        dim_arg = expr_args.pop(0)
        assert isinstance(dim_arg, ir.Var)
        dim_ty = self.typemap[dim_arg.name]
        if isinstance(dim_ty, types.IntegerLiteral):
            dim = dim_ty.literal_value
        else:
            raise RuntimeError(
                f"Expected integer literal for dim, got {dim_ty}"
            )
        expr_args_no_longer_needed.append(dim_arg)

        # Items per thread should be next.
        items_per_thread_arg = expr_args.pop(0)
        assert isinstance(items_per_thread_arg, ir.Var)
        items_per_thread_ty = self.typemap[items_per_thread_arg.name]
        if isinstance(items_per_thread_ty, types.IntegerLiteral):
            items_per_thread = items_per_thread_ty.literal_value
        else:
            msg = (
                "Expected integer literal for items_per_thread, "
                f"got {items_per_thread_ty}"
            )
            raise RuntimeError(msg)
        expr_args_no_longer_needed.append(items_per_thread_arg)

        if expr.kws:
            kws = dict(expr.kws)
            if "algorithm" in kws:
                algorithm = kws["algorithm"]
                assert isinstance(algorithm, ir.Var)
                algorithm_ty = self.typemap[algorithm.name]
                assert isinstance(algorithm_ty, types.EnumMember)
                algorithm_id = algorithm.scope.redefined[int]
                expr_args_no_longer_needed.append(algorithm)

        if algorithm_id is None:
            algorithm_id = int(cs.impl_class.default_algorithm)

        cs.dtype = dtype
        cs.dim = dim
        cs.items_per_thread = items_per_thread
        cs.algorithm_id = algorithm_id
        cs.src = src
        cs.dst = dst
        cs.runtime_args = runtime_args

    def _handle_load_or_store(self, coop_stmt):
        """
        Handle a load or store instruction by replacing it with a call to the
        cooperative implementation.
        """
        cs = coop_stmt
        expr = cs.expr

        impl_class = cs.impl_class

        # Create a global variable for the invocable.
        scope = cs.instr.target.scope
        g_var_name = f"${cs.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        instance = cs.instance = impl_class(
            cs.dtype,
            cs.dim,
            cs.items_per_thread,
            cs.algorithm_id,
        )
        invocable = cs.invocable = instance.invocable

        g_assign = ir.Assign(
            value=ir.Global(g_var_name, invocable, expr.loc),
            target=g_var,
            loc=expr.loc,
        )

        new_call = ir.Expr.call(
            func=g_var,
            args=cs.runtime_args,
            kws=(),
            loc=expr.loc,
        )

        new_expr = ir.Expr(
            op=cs.expr_name,
            loc=expr.loc,
            func=g_var,
            args=cs.runtime_args,
            kws=(),
            coop_stmt=cs,
        )

        new_assign = ir.Assign(
            #value=new_call,
            value=new_expr,
            target=cs.instr.target,
            loc=cs.instr.loc,
        )

        if cs.is_load:
            first = cs.src
            second = cs.dst
        else:
            first = cs.dst
            second = cs.src

        first_ty = self.typemap[first.name]
        second_ty = self.typemap[second.name]

        sig = Signature(
            types.none,
            args=(first_ty, second_ty),
            recvr=None,
            pysig=None,
        )

        self.calltypes[new_expr] = sig

        # Update typemap/calltypes for the new call.

        new_template = make_concrete_template(
            name=f"{impl_class.__name__}_implicit_temp_storage",
            key=invocable,
            signatures=[sig],
        )

        new_template = make_callable_template(
            key=invocable,
            typer=impl_class._typer_implicit_temp_storage,
            recvr=None,
        )


        #func_ty = types.Function(new_callable_template)
        func_ty = types.Function(new_template)

        # I can't imagine this is the correct way to achieve this.
        func_ty._impl_keys = {
            sig.args: invocable,
        }

        existing = self.typemap.get(g_var.name, None)
        if existing:
            raise RuntimeError(
                f"Variable {g_var.name} already exists in typemap."
            )
        self.typemap[g_var.name] = func_ty

        return (g_assign, new_assign)

    def apply(self):
        """
        For each matching coop call, obtain the implementation class and then
        call it with the appropriate arguments.  Inject the `invocable` attr
        of the instance into the block by way of a new global variable and
        assignment, then replace the original call with a call to the global
        variable.
        """
        self.apply_count += 1

        new_block = ir.Block(self.block.scope, self.block.loc)
        new_instrs = []

        unused = set()
        for coop_stmt in self.coop_assigns.values():
            unused.update(coop_stmt.expr_args_no_longer_needed)

        for instr in self.block.body:
            if isinstance(instr, ir.Assign):
                target_name = instr.target.name
                coop_stmt = self.coop_assigns.get(target_name, None)
                if not coop_stmt:
                    if instr.target in unused:
                        continue
                    new_block.append(instr)
                    continue

                if coop_stmt.is_load_or_store:
                    for new_instr in self._handle_load_or_store(coop_stmt):
                        new_instrs.append(new_instr)
                        new_block.append(new_instr)
                else:
                    raise NotImplementedError(
                        f"Don't know how to handle {coop_stmt.template}."
                    )

            elif isinstance(instr, ir.Del):
                if instr in unused:
                    continue
                new_block.append(instr)
                continue

            elif isinstance(instr, ir.Var):
                if instr in unused:
                    continue
                new_block.append(instr)
                continue

            else:
                if instr in unused:
                    continue
                new_block.append(instr)
                continue

        # new_block.dump()
        return new_block
