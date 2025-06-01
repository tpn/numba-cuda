from enum import IntEnum
import operator
from numba.core import errors, types
from numba.core.typing.npydecl import (
    parse_dtype,
    parse_shape,
    register_number_classes,
    register_numpy_ufunc,
    trigonometric_functions,
    comparison_functions,
    math_operations,
    bit_twiddling_functions,
)
from numba.core.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    AbstractTemplate,
    CallableTemplate,
    signature,
    Registry,
)
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function

registry = Registry()
register = registry.register
register_attr = registry.register_attr
register_global = registry.register_global

register_number_classes(register_global)


class Cuda_array_decl(CallableTemplate):
    def generic(self):
        def typer(shape, dtype, alignment=None):
            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any(
                    [not isinstance(s, types.IntegerLiteral) for s in shape]
                ):
                    return None
            else:
                return None

            if alignment is not None:
                permitted = (types.IntegerLiteral, types.NoneType)
                if not isinstance(alignment, permitted):
                    msg = "alignment must be a constant integer"
                    raise errors.RequireLiteralValue(msg)

            # N.B. We don't use alignment for typing; it's not part of
            #      types.Array.  The value supplied to the array declaration
            #      is handled in the lowering.

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout="C")

        return typer


@register
class Cuda_shared_array(Cuda_array_decl):
    key = cuda.shared.array


@register
class Cuda_local_array(Cuda_array_decl):
    key = cuda.local.array


@register
class Cuda_const_array_like(CallableTemplate):
    key = cuda.const.array_like

    def generic(self):
        def typer(ndarray):
            return ndarray

        return typer


@register
class Cuda_threadfence_device(ConcreteTemplate):
    key = cuda.threadfence
    cases = [signature(types.none)]


@register
class Cuda_threadfence_block(ConcreteTemplate):
    key = cuda.threadfence_block
    cases = [signature(types.none)]


@register
class Cuda_threadfence_system(ConcreteTemplate):
    key = cuda.threadfence_system
    cases = [signature(types.none)]


@register
class Cuda_syncwarp(ConcreteTemplate):
    key = cuda.syncwarp
    cases = [signature(types.none), signature(types.none, types.i4)]


@register
class Cuda_vote_sync_intrinsic(ConcreteTemplate):
    key = cuda.vote_sync_intrinsic
    cases = [
        signature(
            types.Tuple((types.i4, types.b1)), types.i4, types.i4, types.b1
        )
    ]


@register
class Cuda_match_any_sync(ConcreteTemplate):
    key = cuda.match_any_sync
    cases = [
        signature(types.i4, types.i4, types.i4),
        signature(types.i4, types.i4, types.i8),
        signature(types.i4, types.i4, types.f4),
        signature(types.i4, types.i4, types.f8),
    ]


@register
class Cuda_match_all_sync(ConcreteTemplate):
    key = cuda.match_all_sync
    cases = [
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.i4),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.i8),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.f4),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.f8),
    ]


@register
class Cuda_activemask(ConcreteTemplate):
    key = cuda.activemask
    cases = [signature(types.uint32)]


@register
class Cuda_lanemask_lt(ConcreteTemplate):
    key = cuda.lanemask_lt
    cases = [signature(types.uint32)]


@register
class Cuda_popc(ConcreteTemplate):
    """
    Supported types from `llvm.popc`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """

    key = cuda.popc
    cases = [
        signature(types.int8, types.int8),
        signature(types.int16, types.int16),
        signature(types.int32, types.int32),
        signature(types.int64, types.int64),
        signature(types.uint8, types.uint8),
        signature(types.uint16, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]


@register
class Cuda_fma(ConcreteTemplate):
    """
    Supported types from `llvm.fma`
    [here](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#standard-c-library-intrinics)
    """

    key = cuda.fma
    cases = [
        signature(types.float32, types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64, types.float64),
    ]


@register
class Cuda_hfma(ConcreteTemplate):
    key = cuda.fp16.hfma
    cases = [
        signature(types.float16, types.float16, types.float16, types.float16)
    ]


@register
class Cuda_cbrt(ConcreteTemplate):
    key = cuda.cbrt
    cases = [
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


@register
class Cuda_brev(ConcreteTemplate):
    key = cuda.brev
    cases = [
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]


@register
class Cuda_clz(ConcreteTemplate):
    """
    Supported types from `llvm.ctlz`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """

    key = cuda.clz
    cases = [
        signature(types.int8, types.int8),
        signature(types.int16, types.int16),
        signature(types.int32, types.int32),
        signature(types.int64, types.int64),
        signature(types.uint8, types.uint8),
        signature(types.uint16, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]


@register
class Cuda_ffs(ConcreteTemplate):
    """
    Supported types from `llvm.cttz`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """

    key = cuda.ffs
    cases = [
        signature(types.uint32, types.int8),
        signature(types.uint32, types.int16),
        signature(types.uint32, types.int32),
        signature(types.uint32, types.int64),
        signature(types.uint32, types.uint8),
        signature(types.uint32, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint32, types.uint64),
    ]


@register
class Cuda_selp(AbstractTemplate):
    key = cuda.selp

    def generic(self, args, kws):
        assert not kws
        test, a, b = args

        # per docs
        # http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp
        supported_types = (
            types.float64,
            types.float32,
            types.int16,
            types.uint16,
            types.int32,
            types.uint32,
            types.int64,
            types.uint64,
        )

        if a != b or a not in supported_types:
            return

        return signature(a, test, a, a)


def _genfp16_unary(l_key):
    @register
    class Cuda_fp16_unary(ConcreteTemplate):
        key = l_key
        cases = [signature(types.float16, types.float16)]

    return Cuda_fp16_unary


def _genfp16_unary_operator(l_key):
    @register_global(l_key)
    class Cuda_fp16_unary(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            if len(args) == 1 and args[0] == types.float16:
                return signature(types.float16, types.float16)

    return Cuda_fp16_unary


def _genfp16_binary(l_key):
    @register
    class Cuda_fp16_binary(ConcreteTemplate):
        key = l_key
        cases = [signature(types.float16, types.float16, types.float16)]

    return Cuda_fp16_binary


@register_global(float)
class Float(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws

        [arg] = args

        if arg == types.float16:
            return signature(arg, arg)


def _genfp16_binary_comparison(l_key):
    @register
    class Cuda_fp16_cmp(ConcreteTemplate):
        key = l_key

        cases = [signature(types.b1, types.float16, types.float16)]

    return Cuda_fp16_cmp


# If multiple ConcreteTemplates provide typing for a single function, then
# function resolution will pick the first compatible typing it finds even if it
# involves inserting a cast that would be considered undesirable (in this
# specific case, float16s could be cast to float32s for comparisons).
#
# To work around this, we instead use an AbstractTemplate that implements
# exactly the casting logic that we desire. The AbstractTemplate gets
# considered in preference to ConcreteTemplates during typing.
#
# This is tracked as Issue #7863 (https://github.com/numba/numba/issues/7863) -
# once this is resolved it should be possible to replace this AbstractTemplate
# with a ConcreteTemplate to simplify the logic.


def _fp16_binary_operator(l_key, retty):
    @register_global(l_key)
    class Cuda_fp16_operator(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws

            if len(args) == 2 and (
                args[0] == types.float16 or args[1] == types.float16
            ):
                if args[0] == types.float16:
                    convertible = self.context.can_convert(args[1], args[0])
                else:
                    convertible = self.context.can_convert(args[0], args[1])

                # We allow three cases here:
                #
                # 1. fp16 to fp16 - Conversion.exact
                # 2. fp16 to other types fp16 can be promoted to
                #  - Conversion.promote
                # 3. fp16 to int8 (safe conversion) -
                #  - Conversion.safe

                if (
                    (convertible == Conversion.exact)
                    or (convertible == Conversion.promote)
                    or (convertible == Conversion.safe)
                ):
                    return signature(retty, types.float16, types.float16)

    return Cuda_fp16_operator


def _genfp16_comparison_operator(op):
    return _fp16_binary_operator(op, types.b1)


def _genfp16_binary_operator(op):
    return _fp16_binary_operator(op, types.float16)


Cuda_hadd = _genfp16_binary(cuda.fp16.hadd)
Cuda_add = _genfp16_binary_operator(operator.add)
Cuda_iadd = _genfp16_binary_operator(operator.iadd)
Cuda_hsub = _genfp16_binary(cuda.fp16.hsub)
Cuda_sub = _genfp16_binary_operator(operator.sub)
Cuda_isub = _genfp16_binary_operator(operator.isub)
Cuda_hmul = _genfp16_binary(cuda.fp16.hmul)
Cuda_mul = _genfp16_binary_operator(operator.mul)
Cuda_imul = _genfp16_binary_operator(operator.imul)
Cuda_hmax = _genfp16_binary(cuda.fp16.hmax)
Cuda_hmin = _genfp16_binary(cuda.fp16.hmin)
Cuda_hneg = _genfp16_unary(cuda.fp16.hneg)
Cuda_neg = _genfp16_unary_operator(operator.neg)
Cuda_habs = _genfp16_unary(cuda.fp16.habs)
Cuda_abs = _genfp16_unary_operator(abs)
Cuda_heq = _genfp16_binary_comparison(cuda.fp16.heq)
_genfp16_comparison_operator(operator.eq)
Cuda_hne = _genfp16_binary_comparison(cuda.fp16.hne)
_genfp16_comparison_operator(operator.ne)
Cuda_hge = _genfp16_binary_comparison(cuda.fp16.hge)
_genfp16_comparison_operator(operator.ge)
Cuda_hgt = _genfp16_binary_comparison(cuda.fp16.hgt)
_genfp16_comparison_operator(operator.gt)
Cuda_hle = _genfp16_binary_comparison(cuda.fp16.hle)
_genfp16_comparison_operator(operator.le)
Cuda_hlt = _genfp16_binary_comparison(cuda.fp16.hlt)
_genfp16_comparison_operator(operator.lt)
_genfp16_binary_operator(operator.truediv)
_genfp16_binary_operator(operator.itruediv)


def _resolve_wrapped_unary(fname):
    link = tuple()
    decl = declare_device_function(
        f"__numba_wrapper_{fname}",
        types.float16,
        (types.float16,),
        link,
        use_cooperative=False,
    )
    return types.Function(decl)


def _resolve_wrapped_binary(fname):
    link = tuple()
    decl = declare_device_function(
        f"__numba_wrapper_{fname}",
        types.float16,
        (
            types.float16,
            types.float16,
        ),
        link,
        use_cooperative=False,
    )
    return types.Function(decl)


hsin_device = _resolve_wrapped_unary("hsin")
hcos_device = _resolve_wrapped_unary("hcos")
hlog_device = _resolve_wrapped_unary("hlog")
hlog10_device = _resolve_wrapped_unary("hlog10")
hlog2_device = _resolve_wrapped_unary("hlog2")
hexp_device = _resolve_wrapped_unary("hexp")
hexp10_device = _resolve_wrapped_unary("hexp10")
hexp2_device = _resolve_wrapped_unary("hexp2")
hsqrt_device = _resolve_wrapped_unary("hsqrt")
hrsqrt_device = _resolve_wrapped_unary("hrsqrt")
hfloor_device = _resolve_wrapped_unary("hfloor")
hceil_device = _resolve_wrapped_unary("hceil")
hrcp_device = _resolve_wrapped_unary("hrcp")
hrint_device = _resolve_wrapped_unary("hrint")
htrunc_device = _resolve_wrapped_unary("htrunc")
hdiv_device = _resolve_wrapped_binary("hdiv")


# generate atomic operations
def _gen(l_key, supported_types):
    @register
    class Cuda_atomic(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            ary, idx, val = args

            if ary.dtype not in supported_types:
                return

            if ary.ndim == 1:
                return signature(ary.dtype, ary, types.intp, ary.dtype)
            elif ary.ndim > 1:
                return signature(ary.dtype, ary, idx, ary.dtype)

    return Cuda_atomic


all_numba_types = (
    types.float64,
    types.float32,
    types.int32,
    types.uint32,
    types.int64,
    types.uint64,
)

integer_numba_types = (types.int32, types.uint32, types.int64, types.uint64)

unsigned_int_numba_types = (types.uint32, types.uint64)

Cuda_atomic_add = _gen(cuda.atomic.add, all_numba_types)
Cuda_atomic_sub = _gen(cuda.atomic.sub, all_numba_types)
Cuda_atomic_max = _gen(cuda.atomic.max, all_numba_types)
Cuda_atomic_min = _gen(cuda.atomic.min, all_numba_types)
Cuda_atomic_nanmax = _gen(cuda.atomic.nanmax, all_numba_types)
Cuda_atomic_nanmin = _gen(cuda.atomic.nanmin, all_numba_types)
Cuda_atomic_and = _gen(cuda.atomic.and_, integer_numba_types)
Cuda_atomic_or = _gen(cuda.atomic.or_, integer_numba_types)
Cuda_atomic_xor = _gen(cuda.atomic.xor, integer_numba_types)
Cuda_atomic_inc = _gen(cuda.atomic.inc, unsigned_int_numba_types)
Cuda_atomic_dec = _gen(cuda.atomic.dec, unsigned_int_numba_types)
Cuda_atomic_exch = _gen(cuda.atomic.exch, integer_numba_types)


@register
class Cuda_atomic_compare_and_swap(AbstractTemplate):
    key = cuda.atomic.compare_and_swap

    def generic(self, args, kws):
        assert not kws
        ary, old, val = args
        dty = ary.dtype

        if dty in integer_numba_types and ary.ndim == 1:
            return signature(dty, ary, dty, dty)


@register
class Cuda_atomic_cas(AbstractTemplate):
    key = cuda.atomic.cas

    def generic(self, args, kws):
        assert not kws
        ary, idx, old, val = args
        dty = ary.dtype

        if dty not in integer_numba_types:
            return

        if ary.ndim == 1:
            return signature(dty, ary, types.intp, dty, dty)
        elif ary.ndim > 1:
            return signature(dty, ary, idx, dty, dty)


@register_global(breakpoint)
class Cuda_breakpoint(ConcreteTemplate):
    cases = [signature(types.none)]


@register
class Cuda_nanosleep(ConcreteTemplate):
    key = cuda.nanosleep

    cases = [signature(types.void, types.uint32)]


@register_attr
class Dim3_attrs(AttributeTemplate):
    key = dim3

    def resolve_x(self, mod):
        return types.int32

    def resolve_y(self, mod):
        return types.int32

    def resolve_z(self, mod):
        return types.int32


@register_attr
class CudaSharedModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.shared)

    def resolve_array(self, mod):
        return types.Function(Cuda_shared_array)


@register_attr
class CudaConstModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.const)

    def resolve_array_like(self, mod):
        return types.Function(Cuda_const_array_like)


@register_attr
class CudaLocalModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.local)

    def resolve_array(self, mod):
        return types.Function(Cuda_local_array)


@register_attr
class CudaAtomicTemplate(AttributeTemplate):
    key = types.Module(cuda.atomic)

    def resolve_add(self, mod):
        return types.Function(Cuda_atomic_add)

    def resolve_sub(self, mod):
        return types.Function(Cuda_atomic_sub)

    def resolve_and_(self, mod):
        return types.Function(Cuda_atomic_and)

    def resolve_or_(self, mod):
        return types.Function(Cuda_atomic_or)

    def resolve_xor(self, mod):
        return types.Function(Cuda_atomic_xor)

    def resolve_inc(self, mod):
        return types.Function(Cuda_atomic_inc)

    def resolve_dec(self, mod):
        return types.Function(Cuda_atomic_dec)

    def resolve_exch(self, mod):
        return types.Function(Cuda_atomic_exch)

    def resolve_max(self, mod):
        return types.Function(Cuda_atomic_max)

    def resolve_min(self, mod):
        return types.Function(Cuda_atomic_min)

    def resolve_nanmin(self, mod):
        return types.Function(Cuda_atomic_nanmin)

    def resolve_nanmax(self, mod):
        return types.Function(Cuda_atomic_nanmax)

    def resolve_compare_and_swap(self, mod):
        return types.Function(Cuda_atomic_compare_and_swap)

    def resolve_cas(self, mod):
        return types.Function(Cuda_atomic_cas)


@register_attr
class CudaFp16Template(AttributeTemplate):
    key = types.Module(cuda.fp16)

    def resolve_hadd(self, mod):
        return types.Function(Cuda_hadd)

    def resolve_hsub(self, mod):
        return types.Function(Cuda_hsub)

    def resolve_hmul(self, mod):
        return types.Function(Cuda_hmul)

    def resolve_hdiv(self, mod):
        return hdiv_device

    def resolve_hneg(self, mod):
        return types.Function(Cuda_hneg)

    def resolve_habs(self, mod):
        return types.Function(Cuda_habs)

    def resolve_hfma(self, mod):
        return types.Function(Cuda_hfma)

    def resolve_hsin(self, mod):
        return hsin_device

    def resolve_hcos(self, mod):
        return hcos_device

    def resolve_hlog(self, mod):
        return hlog_device

    def resolve_hlog10(self, mod):
        return hlog10_device

    def resolve_hlog2(self, mod):
        return hlog2_device

    def resolve_hexp(self, mod):
        return hexp_device

    def resolve_hexp10(self, mod):
        return hexp10_device

    def resolve_hexp2(self, mod):
        return hexp2_device

    def resolve_hfloor(self, mod):
        return hfloor_device

    def resolve_hceil(self, mod):
        return hceil_device

    def resolve_hsqrt(self, mod):
        return hsqrt_device

    def resolve_hrsqrt(self, mod):
        return hrsqrt_device

    def resolve_hrcp(self, mod):
        return hrcp_device

    def resolve_hrint(self, mod):
        return hrint_device

    def resolve_htrunc(self, mod):
        return htrunc_device

    def resolve_heq(self, mod):
        return types.Function(Cuda_heq)

    def resolve_hne(self, mod):
        return types.Function(Cuda_hne)

    def resolve_hge(self, mod):
        return types.Function(Cuda_hge)

    def resolve_hgt(self, mod):
        return types.Function(Cuda_hgt)

    def resolve_hle(self, mod):
        return types.Function(Cuda_hle)

    def resolve_hlt(self, mod):
        return types.Function(Cuda_hlt)

    def resolve_hmax(self, mod):
        return types.Function(Cuda_hmax)

    def resolve_hmin(self, mod):
        return types.Function(Cuda_hmin)


# cuda.cooperative (block, warp)


class BlockLoadAlgorithm(IntEnum):
    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


class WarpLoadAlgorithm(IntEnum):
    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3


class BlockStoreAlgorithm(IntEnum):
    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


class WarpStoreAlgorithm(IntEnum):
    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3


class BlockScanAlgorithm(IntEnum):
    RAKING = 0
    RAKING_MEMOIZE = 1
    WARP_SCAN = 2


class BlockReduceAlgorithm(IntEnum):
    RAKING_COMMUTATIVE_ONLY = 0
    RAKING = 1
    WARP_REDUCTIONS = 2


cuda.BlockLoadAlgorithm = BlockLoadAlgorithm
cuda.WarpLoadAlgorithm = WarpLoadAlgorithm
cuda.BlockStoreAlgorithm = BlockStoreAlgorithm
cuda.WarpStoreAlgorithm = WarpStoreAlgorithm
cuda.BlockScanAlgorithm = BlockScanAlgorithm
cuda.BlockReduceAlgorithm = BlockReduceAlgorithm

# Dummy sentinel used to detect missing arguments.
_MISSING_SENTINEL = object()


def _is_src_first(primitive_name):
    return primitive_name.endswith(".load")


def _bind_and_validate_src_dst(args, kwds, primitive_name):
    """
    Bind the user-supplied *src* and *dst* arguments (positional or keyword)
    and validate that

        * both are supplied and are device arrays,
        * the same name is not provided twice (pos+kw),
        * their dtype / ndim / layout agree (layout check is skipped if either
          array has layout 'A' = any).

    This function modifies args and kwds by removing consumed arguments.

    Parameters
    ----------
    args : list
        Positional arguments that reached the typer (will be modified).
    kwds : dict
        Keyword arguments that reached the typer (will be modified).
    primitive_name : str
        e.g. "cuda.block.load" – used only in diagnostics.

    Returns
    -------
    (src_type, dst_type) : Tuple[numba.types.Array, numba.types.Array]
        The Numba *types* for src and dst after binding.
    """

    # ------------------------------------------------------------------
    # 1.  Detect whether this primitive is (src, dst) or (dst, src)
    # ------------------------------------------------------------------
    src_first = _is_src_first(primitive_name)

    # Mapping of positional slots to names:
    #   load : arg0→src , arg1→dst
    #   store: arg0→dst , arg1→src
    pos_names = ("src", "dst") if src_first else ("dst", "src")

    # ------------------------------------------------------------------
    # 2.  Check for duplicate specification (positional *and* keyword)
    # ------------------------------------------------------------------
    for idx, name in enumerate(pos_names):
        if name in kwds and len(args) > idx:
            raise errors.TypingError(
                f"{primitive_name}: '{name}' specified both positionally "
                "and as a keyword"
            )

    # ------------------------------------------------------------------
    # 3.  Bind src / dst from args or kwds
    # ------------------------------------------------------------------
    src = None
    dst = None

    # Try to get from positional args first.
    if src_first:
        if len(args) >= 1:
            src = args.pop(0)
        if len(args) >= 1:
            dst = args.pop(0)
    else:  # store primitives
        if len(args) >= 1:
            dst = args.pop(0)
        if len(args) >= 1:
            src = args.pop(0)

    # Get from keywords if not found in positional
    if src is None and "src" in kwds:
        src = kwds.pop("src")
    if dst is None and "dst" in kwds:
        dst = kwds.pop("dst")

    # ------------------------------------------------------------------
    # 4.  Presence check
    # ------------------------------------------------------------------
    if src is None or dst is None:
        raise errors.TypingError(
            f"{primitive_name} needs both 'src' and 'dst' arrays"
        )

    # ------------------------------------------------------------------
    # 5.  Type and structural compatibility checks
    # ------------------------------------------------------------------
    if not isinstance(src, types.Array) or not isinstance(dst, types.Array):
        raise errors.TypingError(
            f"{primitive_name} requires both 'src' and 'dst' to be device "
            "arrays"
        )

    # dtype
    if src.dtype != dst.dtype:
        raise errors.TypingError(
            f"{primitive_name} requires 'src' and 'dst' to have the same "
            f"dtype (got {src.dtype} vs {dst.dtype})"
        )

    # ndim
    if src.ndim != dst.ndim:
        raise errors.TypingError(
            f"{primitive_name} requires 'src' and 'dst' to have the same "
            f"number of dimensions (got {src.ndim} vs {dst.ndim})"
        )

    # layout – skip if either is 'A' (unknown/any)
    if src.layout != "A" and dst.layout != "A" and src.layout != dst.layout:
        raise errors.TypingError(
            f"{primitive_name} requires 'src' and 'dst' to have the same "
            f"layout (got {src.layout!r} vs {dst.layout!r})"
        )

    return (src, dst)


def _as_pos_integer_literal_old2(val, name, primitive):
    """If *val* is Integer(Literal) verify positivity and, if possible,
    promote to IntegerLiteral to keep the information that it is a constant."""
    if isinstance(val, types.IntegerLiteral):
        if val.literal_value <= 0:
            raise errors.TypingError(
                f"'{name}' must be a positive integer; got {val.literal_value}"
            )
        # Normalize to intp instead of preserving the literal type
        return types.intp

    if isinstance(val, types.Integer):
        # Compile-time constants sometimes arrive as plain Integer types
        lit = getattr(val, "literal_value", None)
        if lit is not None:  # we know the exact value
            if lit <= 0:
                raise errors.TypingError(
                    f"'{name}' must be a positive integer; got {lit}"
                )
            # Normalize to intp instead of creating IntegerLiteral
            return types.intp
        # run-time scalar – positivity will be checked later in lowering
        return val

    # Anything else isn't an integer scalar
    raise errors.TypingError(f"{primitive}: '{name}' must be an integer scalar")


def _as_pos_integer_literal(val, name, primitive):
    # IntegerLiteral ➜ keep as-is (after the >0 check)
    if isinstance(val, types.IntegerLiteral):
        if val.literal_value <= 0:
            raise errors.TypingError(
                f"'{name}' must be a positive integer; got {val.literal_value}"
            )
        return val

    # Literal(int) ➜ keep as-is
    if isinstance(val, types.Literal):
        if isinstance(val.literal_value, int):
            if val.literal_value <= 0:
                raise errors.TypingError(
                    f"'{name}' must be a positive integer; got {val.literal_value}"
                )
            return val

    # Plain Integer ➜ keep as-is (positivity checked at run-time in lowering)
    if isinstance(val, types.Integer):
        return val

    raise errors.TypingError(f"{primitive}: '{name}' must be an integer scalar")


def _validate_positive_int_literal(
    args, kwds, value, param_name: str, primitive_name: str
):
    """
    Fetch *param_name* from the call, detect duplicates, require presence,
    and return *exactly* the Numba type that was supplied (possibly promoted
    to IntegerLiteral).
    """
    # 1. keyword beats positional duplicates
    if param_name in kwds:
        if value is not _MISSING_SENTINEL:
            raise errors.TypingError(
                f"{primitive_name}: '{param_name}' specified both "
                "positionally and as a keyword"
            )
        value = kwds.pop(param_name)

    # 2. still unset?  pop from *args*
    if value is _MISSING_SENTINEL and args:
        value = args.pop(0)

    # 3. presence check
    if value is _MISSING_SENTINEL:
        raise errors.TypingError(f"{primitive_name} requires '{param_name}'")

    # 4. type / sign check – and possible promotion to IntegerLiteral
    return _as_pos_integer_literal(value, param_name, primitive_name)


def _validate_positive_int_literal_old(
    args,
    kwds,
    value,
    param_name: str,
    primitive_name: str,
):
    """
    Fetch *param_name* from either *value* (positional), *args*, or *kwds*,
    check that the user didn't supply it twice, and verify the type / sign.

    On return:
      * The corresponding entry is popped from *args* / *kwds*.
      * The returned value is a Numba *type* that is a subtype of
        `types.Integer` (possibly `types.IntegerLiteral`).
    """
    # 1.  Pull from keyword dict (and watch for duplicates)
    if param_name in kwds:
        if value is not _MISSING_SENTINEL:
            raise errors.TypingError(
                f"{primitive_name}: '{param_name}' specified both "
                "positionally and as a keyword"
            )
        value = kwds.pop(param_name)

    # 2.  Pull from positional list if still unset
    if value is _MISSING_SENTINEL and args:
        value = args.pop(0)

    # 3.  Presence check
    if value is _MISSING_SENTINEL:
        raise errors.TypingError(f"{primitive_name} requires '{param_name}'")

    # 4.  Type and positivity checks
    if isinstance(value, types.IntegerLiteral):
        if value.literal_value <= 0:
            raise errors.TypingError(
                f"'{param_name}' must be a positive integer; "
                f"got {value.literal_value}"
            )
        return value  # literal, keep as-is

    if isinstance(value, types.Literal):
        pyval = value.literal_value
        if isinstance(pyval, int):
            if pyval <= 0:
                raise errors.TypingError(
                    f"'{param_name}' must be a positive integer; got {pyval}"
                )
            return types.IntegerLiteral(pyval)

    if not isinstance(value, types.Integer):
        raise errors.TypingError(
            f"{primitive_name}: '{param_name}' must be an integer scalar"
        )

    # If the integer happens to have a literal_value attribute (compile-time
    # constant folded by Numba), check its sign and turn it into a literal
    # type – this lets constant-propagation hit the fast path.
    if hasattr(value, "literal_value"):
        lit = value.literal_value
        if lit <= 0:
            raise errors.TypingError(
                f"'{param_name}' must be a positive integer; got {lit}"
            )
        return types.IntegerLiteral(lit)

    # Generic integer (run-time value) – handed off to lowering for the final
    # constant-ness check.
    return types.intp


def _validate_threads_per_block(args, kwds, current, primitive_name: str):
    return _validate_positive_int_literal(
        args, kwds, current, "threads_per_block", primitive_name
    )


def _validate_items_per_thread(args, kwds, current, primitive_name: str):
    return _validate_positive_int_literal(
        args, kwds, current, "items_per_thread", primitive_name
    )


def _validate_algorithm(args, kwds, algorithm, primitive_name: str, enum_cls):
    """
    Make sure *algorithm* is a literal belonging to the given enum class.

    This function modifies args and kwds by removing consumed arguments.

    Returns the validated algorithm value.
    """
    original_algorithm = algorithm

    # Check if algorithm is in kwds
    if "algorithm" in kwds:
        if algorithm is not _MISSING_SENTINEL:
            raise errors.TypingError(
                f"{primitive_name}: 'algorithm' specified both positionally "
                "and as a keyword"
            )
        algorithm = kwds.pop("algorithm")

    # If still missing and we have positional args, try to get from there
    if algorithm is _MISSING_SENTINEL:
        if args:
            algorithm = args.pop(0)
        else:
            # If no algorithm was specified, use the default
            return None

    enum_name = enum_cls.__name__
    user_facing_name = f"cuda.{enum_name}"

    if not isinstance(algorithm, types.EnumMember):
        msg = (
            f"algorithm for {primitive_name} must be a member "
            f"of {user_facing_name}, got {original_algorithm}"
        )
        raise errors.TypingError(msg)

    if algorithm.instance_class is not enum_cls:
        name = algorithm.instance_class.__name__
        msg = (
            f"algorithm for {primitive_name} must be a member "
            f"of {user_facing_name}, got {name} "
        )
        raise errors.TypingError(msg)

    return algorithm


class Coop_load_store_decl(CallableTemplate):
    """
    Base class for all cooperative load and store functions.  Subclasses must
    define the following attributes:
      - key: the function name (e.g. cuda.block.load)
      - primitive_name: the name of the primitive (e.g. "cuda.block.load")
      - algorithm_enum: the enum class for the algorithm (e.g.
        BlockLoadAlgorithm)
      - default_algorithm: the default algorithm to use if not specified
    """

    def generic(self):
        print(f"Entered generic for {self.key}, {self.primitive_name}")

        def typer(
            *args,
            threads_per_block=_MISSING_SENTINEL,
            items_per_thread=_MISSING_SENTINEL,
            algorithm=_MISSING_SENTINEL,
            **kwds,
        ):
            # Convert args to a mutable list and make a copy of kwds
            args = list(args)
            kwds = dict(kwds)

            (src, dst) = _bind_and_validate_src_dst(
                args, kwds, self.primitive_name
            )

            threads_per_block = _validate_threads_per_block(
                args,
                kwds,
                threads_per_block,
                self.primitive_name,
            )

            items_per_thread = _validate_items_per_thread(
                args,
                kwds,
                items_per_thread,
                self.primitive_name,
            )

            algorithm = _validate_algorithm(
                args,
                kwds,
                algorithm,
                self.primitive_name,
                self.algorithm_enum,
            )

            # If args or kwds still have values, the user has passed extra
            # arguments that we don't support.
            if args:
                raise errors.TypingError(
                    f"{self.primitive_name} does not support additional "
                    f"positional arguments: {', '.join(map(str, args))}"
                )
            if kwds:
                names = ", ".join(kwds.keys())
                raise errors.TypingError(
                    f"{self.primitive_name} does not support additional "
                    f"keyword arguments: {names}"
                )

            if _is_src_first(self.primitive_name):
                array_args = (src, dst)
            else:
                array_args = (dst, src)

            arglist = [
                *array_args,
                threads_per_block,
                items_per_thread,
            ]

            if algorithm is not None:
                arglist.append(algorithm)
            else:
                arglist.append(self.default_algorithm.value)

            return types.VarArg(types.Any)

            # Doesn't work:
            return signature(types.void, types.VarArg(types.Any))

            # Doesn't work:
            return signature(
                types.void,
                *arglist,
            )

        return typer


@register
class Coop_block_load(Coop_load_store_decl):
    key = cuda.block.load
    primitive_name = "cuda.block.load"
    algorithm_enum = BlockLoadAlgorithm
    default_algorithm = BlockLoadAlgorithm.DIRECT


@register
class Coop_warp_load(Coop_load_store_decl):
    key = cuda.warp.load
    primitive_name = "cuda.warp.load"
    algorithm_enum = WarpLoadAlgorithm
    default_algorithm = WarpLoadAlgorithm.DIRECT


@register
class Coop_block_store(Coop_load_store_decl):
    key = cuda.block.store
    primitive_name = "cuda.block.store"
    algorithm_enum = BlockStoreAlgorithm
    default_algorithm = BlockStoreAlgorithm.DIRECT


@register
class Coop_warp_store(Coop_load_store_decl):
    key = cuda.warp.store
    primitive_name = "cuda.warp.store"
    algorithm_enum = WarpStoreAlgorithm
    default_algorithm = WarpStoreAlgorithm.DIRECT


@register_attr
class CudaBlockModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.block)

    def resolve_load(self, mod):
        return types.Function(Coop_block_load)

    def resolve_store(self, mod):
        return types.Function(Coop_block_store)


@register_attr
class CudaWarpModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.warp)

    def resolve_load(self, mod):
        return types.Function(Coop_warp_load)

    def resolve_store(self, mod):
        return types.Function(Coop_warp_store)


@register_attr
class CudaModuleTemplate(AttributeTemplate):
    key = types.Module(cuda)

    # cuda.cooperative: begin

    def resolve_block(self, mod):
        return types.Module(cuda.block)

    def resolve_warp(self, mod):
        return types.Module(cuda.warp)

    def resolve_BlockLoadAlgorithm(self, mod):
        return types.Module(BlockLoadAlgorithm)

    def resolve_BlockStoreAlgorithm(self, mod):
        return types.Module(BlockStoreAlgorithm)

    # cuda.cooperative: end

    def resolve_cg(self, mod):
        return types.Module(cuda.cg)

    def resolve_threadIdx(self, mod):
        return dim3

    def resolve_blockIdx(self, mod):
        return dim3

    def resolve_blockDim(self, mod):
        return dim3

    def resolve_gridDim(self, mod):
        return dim3

    def resolve_laneid(self, mod):
        return types.int32

    def resolve_shared(self, mod):
        return types.Module(cuda.shared)

    def resolve_popc(self, mod):
        return types.Function(Cuda_popc)

    def resolve_brev(self, mod):
        return types.Function(Cuda_brev)

    def resolve_clz(self, mod):
        return types.Function(Cuda_clz)

    def resolve_ffs(self, mod):
        return types.Function(Cuda_ffs)

    def resolve_fma(self, mod):
        return types.Function(Cuda_fma)

    def resolve_cbrt(self, mod):
        return types.Function(Cuda_cbrt)

    def resolve_threadfence(self, mod):
        return types.Function(Cuda_threadfence_device)

    def resolve_threadfence_block(self, mod):
        return types.Function(Cuda_threadfence_block)

    def resolve_threadfence_system(self, mod):
        return types.Function(Cuda_threadfence_system)

    def resolve_syncwarp(self, mod):
        return types.Function(Cuda_syncwarp)

    def resolve_vote_sync_intrinsic(self, mod):
        return types.Function(Cuda_vote_sync_intrinsic)

    def resolve_match_any_sync(self, mod):
        return types.Function(Cuda_match_any_sync)

    def resolve_match_all_sync(self, mod):
        return types.Function(Cuda_match_all_sync)

    def resolve_activemask(self, mod):
        return types.Function(Cuda_activemask)

    def resolve_lanemask_lt(self, mod):
        return types.Function(Cuda_lanemask_lt)

    def resolve_selp(self, mod):
        return types.Function(Cuda_selp)

    def resolve_nanosleep(self, mod):
        return types.Function(Cuda_nanosleep)

    def resolve_atomic(self, mod):
        return types.Module(cuda.atomic)

    def resolve_fp16(self, mod):
        return types.Module(cuda.fp16)

    def resolve_const(self, mod):
        return types.Module(cuda.const)

    def resolve_local(self, mod):
        return types.Module(cuda.local)


register_global(cuda, types.Module(cuda))


# NumPy

for func in trigonometric_functions:
    register_numpy_ufunc(func, register_global)

for func in comparison_functions:
    register_numpy_ufunc(func, register_global)

for func in bit_twiddling_functions:
    register_numpy_ufunc(func, register_global)

for func in math_operations:
    if func in ("log", "log2", "log10"):
        register_numpy_ufunc(func, register_global)
