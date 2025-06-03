# cuda.cooperative

import inspect
from numba.core.typing.templates import Signature

# coop_rewrite.py  –  put this on your PYTHONPATH
from numba.core import ir, types, ir_utils
from numba.core.rewrites import register_rewrite, Rewrite
from numba import cuda

from . import block

@register_rewrite('after-inference')
class InterceptCooperativeCalls(Rewrite):
    """
    Stage-5a pass that intercepts every call to any
    cuda.<block|warp>.<load|store> intrinsic.

    * `match()` is run once per basic-block – keep it cheap.
    * `apply()` is only entered if `match()` returned True.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from ..cudadecl import (
            Coop_block_load,
            Coop_block_store,
            #Coop_warp_load,
            #Coop_warp_store,
        )

        self._COOP_TEMPLATES = {
            Coop_block_load: block.load,
            Coop_block_store: block.store,
            #Coop_warp_load: warp.load,
            #Coop_warp_store: warp.store,
        }

    def match(self, func_ir, block, typemap, calltypes, **kw):
        self._targets = []
        first = True

        for stmt in block.body:
            if not isinstance(stmt, ir.Assign):
                continue
            expr = stmt.value
            if not (isinstance(expr, ir.Expr) and expr.op == 'call'):
                continue

            cty = calltypes[expr]
            assert isinstance(cty, Signature)

            func = typemap[expr.func.name]

            templates = func.templates
            assert len(templates) == 1, templates
            template = templates[0]

            if template not in self._COOP_TEMPLATES:
                continue

            if first:
                self.func_ir = func_ir
                self.block = block
                self.typemap = typemap
                self.calltypes = calltypes
                first = False

            self._targets.append((stmt, expr, template))

        return bool(self._targets)

    def apply(self):
        """
        Do whatever you need:
          * rewrite               (replace the Expr with something else)
          * decorate with metadata
          * record statistics
          * dump debugging info
        """
        for (stmt, expr, template) in self._targets:

            impl_class = self._COOP_TEMPLATES[template]

            sig = inspect.signature(impl_class)

            # Figure out how to call the implementation class, then call it.
            # That'll produce an object that'll have an `invocable` attribute,
            # and we want to substitute the call with that.

            msg = (f"[coop-rewrite] {template.__name__} at "
                   f"{expr.loc.filename}:{expr.loc.line}: "
                   f"stmt: {stmt}, expr: {expr}, "
                   f"template: {template}, sig: {sig}")
            print(msg)

            class_name = impl_class.__name__
            is_load = class_name == "load"
            is_store = class_name == "store"
            is_load_or_store = is_load or is_store
            if is_load_or_store:
                dtype = None
                dim = None
                items_per_thread = None
                algorithm = None
                algorithm_id = None
                expr_args = list(expr.args)
                if is_load:
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
                dim = expr_args.pop(0)
                assert isinstance(dim, ir.Var)
                dim_ty = self.typemap[dim.name]
                if isinstance(dim_ty, types.IntegerLiteral):
                    dim = dim_ty.literal_value
                else:
                    raise RuntimeError(
                        f"Expected integer literal for dim, got {dim_ty}"
                    )

                # Items per thread should be next.
                items_per_thread = expr_args.pop(0)
                assert isinstance(items_per_thread, ir.Var)
                items_per_thread_ty = self.typemap[items_per_thread.name]
                if isinstance(items_per_thread_ty, types.IntegerLiteral):
                    items_per_thread = items_per_thread_ty.literal_value
                else:
                    msg = (
                        'Expected integer literal for items_per_thread, '
                        f'got {items_per_thread_ty}'
                    )
                    raise RuntimeError(msg)

                # If there's an arg left, it's the algorithm.
                if expr_args:
                    algorithm = expr_args.pop(0)
                    assert isinstance(algorithm, ir.Var)
                    algorithm_ty = self.typemap[algorithm.name]
                    if isinstance(algorithm_ty, types.EnumMember):
                        algorithm = algorithm_ty.value
                    else:
                        raise RuntimeError(
                            f'Expected enum member for algorithm, '
                            f'got {algorithm_ty}'
                        )
                elif expr.kws:
                    kws = dict(expr.kws)
                    if 'algorithm' in kws:
                        algorithm = kws['algorithm']
                        assert isinstance(algorithm, ir.Var)
                        algorithm_ty = self.typemap[algorithm.name]
                        assert isinstance(algorithm_ty, types.EnumMember)
                        algorithm_id = algorithm.scope.redefined[int]
                        if isinstance(algorithm_ty, types.EnumMember):
                            # Can we get the value at this point?
                            #algorithm = algorithm_ty.value
                            pass
                        else:
                            raise RuntimeError(
                                f'Expected enum member for algorithm, '
                                f'got {algorithm_ty}'
                            )

                if algorithm_id is None:
                    algorithm_id = int(impl_class.default_algorithm)

                impl = impl_class(dtype, dim, items_per_thread, algorithm_id)
                invocable = impl.invocable

                scope = stmt.target.scope
                g_var_name = ir_utils.mk_unique_var(f'${template.__name__}')
                g_var = ir.Var(scope, g_var_name, expr.loc)
                g_assign = ir.Assign(
                    value=ir.Global(g_var_name, invocable, expr.loc),
                    target=g_var,
                    loc=expr.loc,
                )

                # Replace the original call with a call to invocable with
                # the runtime arguments, e.g. ``invocable(src, dst)``.
                new_call = ir.Expr.call(
                    func=g_var,
                    args=runtime_args,
                    kws=(),
                    loc=expr.loc,
                )
                new_assign = ir.Assign(
                    value=new_call,
                    target=stmt.target,
                    loc=stmt.loc,
                )

                # Splice into the block (i.e. replace in-situ).
                body = self.block.body
                idx = body.index(stmt)
                body[idx:idx+1] = [g_assign, new_assign]

                func_ty = types.Function(invocable)
                self.typemap[g_var.name] = func_ty

                if is_load:
                    first = src
                    second = dst
                else:
                    first = dst
                    second = src

                first_ty = self.typemap[first.name]
                second_ty = self.typemap[second.name]

                self.calltypes[new_call] = Signature(
                    types.void,
                    first_ty,
                    second_ty,
                )


        return self.block


LOAD_STORE_SIGNATURE1 = (
    types.Array,  # src or dst
    types.Array,  # dst or src
    types.Integer,  # threads_per_block
    types.Integer,  # items_per_thread
)

LOAD_STORE_SIGNATURE2 = (
    types.Array,  # src or dst
    types.Array,  # dst or src
    types.Integer,  # threads_per_block
    types.Integer,  # items_per_thread
    types.EnumMember,  # algorithm
)

# @lower(cuda.block.load, *LOAD_STORE_SIGNATURE1)
# @lower(cuda.block.load, *LOAD_STORE_SIGNATURE2)


#@lower(cuda.block.load, types.VarArg(types.Any))
#def lower_block_load(context, builder, sig, args):
#    # import ipdb
#    # ipdb.set_trace()
#
#    from .cudadecl import Coop_block_load
#
#    if len(args) == 4:
#        algorithm = Coop_block_load.default_algorithm.value
#        (src_ty, dst_ty, threads_per_block_ty, items_per_thread_ty) = sig.args
#        (src, dst, threads_per_block, items_per_thread) = args
#    else:
#        (src_ty, dst_ty, threads_per_block_ty, items_per_thread_ty, algo_ty) = (
#            sig.args
#        )
#        (src, dst, threads_per_block, items_per_thread, algorithm) = args
#
#    print("ENTERED cuda.block.load LOWERING")
#
#    # Print all the types.
#    print(
#        f"src_ty: {src_ty}, dst_ty: {dst_ty}, "
#        f"threads_per_block_ty: {threads_per_block_ty}, "
#        f"items_per_thread_ty: {items_per_thread_ty}"
#    )
#    # Print all the values.
#    print(
#        f"src: {src}, dst: {dst}, "
#        f"threads_per_block: {threads_per_block}, "
#        f"items_per_thread: {items_per_thread}, "
#        f"algorithm: {algorithm}"
#    )
#    print(f"ENTER LOWER BLOCK LOAD: {algorithm}")
#
#
#@lower(cuda.block.store, types.VarArg(types.Any))
#def lower_block_store(context, builder, sig, args):
#    from .cudadecl import Coop_block_store
#
#    if len(args) == 4:
#        algorithm = Coop_block_store.default_algorithm.value
#        (src_ty, dst_ty, threads_per_block_ty, items_per_thread_ty) = sig.args
#        (src, dst, threads_per_block, items_per_thread) = args
#    else:
#        (src_ty, dst_ty, threads_per_block_ty, items_per_thread_ty, algo_ty) = (
#            sig.args
#        )
#        (src, dst, threads_per_block, items_per_thread, algorithm) = args
#
#    print("ENTER cuda.block.store LOWERING")
#
#    # Print all the types.
#    print(
#        f"src_ty: {src_ty}, dst_ty: {dst_ty}, "
#        f"threads_per_block_ty: {threads_per_block_ty}, "
#        f"items_per_thread_ty: {items_per_thread_ty}"
#    )
#
#    # Print all the values.
#    print(
#        f"src: {src}, dst: {dst}, "
#        f"threads_per_block: {threads_per_block}, "
#        f"items_per_thread: {items_per_thread}, "
#        f"algorithm: {algorithm}"
#    )
