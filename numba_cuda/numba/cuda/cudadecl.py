from collections import defaultdict
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

#
class BlockScanAlgorithm(IntEnum):
    RAKING = 0
    RAKING_MEMOIZE = 1
    WARP_SCAN = 2


class BlockReduceAlgorithm(IntEnum):
    RAKING_COMMUTATIVE_ONLY = 0
    RAKING = 1
    WARP_REDUCTIONS = 2

class ScanMode(IntEnum):
    EXCLUSIVE = 0
    INCLUSIVE = 1

cuda.BlockLoadAlgorithm = BlockLoadAlgorithm
cuda.WarpLoadAlgorithm = WarpLoadAlgorithm
cuda.BlockStoreAlgorithm = BlockStoreAlgorithm
cuda.WarpStoreAlgorithm = WarpStoreAlgorithm
cuda.BlockScanAlgorithm = BlockScanAlgorithm
cuda.BlockReduceAlgorithm = BlockReduceAlgorithm


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

    unsafe_casting = False
    exact_match_required = True
    prefer_literal = True

    def __init__(self, *args, **kwds):
        self._apply_count = 0
        super().__init__(*args, **kwds)

    def apply(self, args, kws):
        self._apply_count += 1
        msg = (
            f"{self.__class__.__name__}.apply("
            f"{self._apply_count}) for "
            f"{self.key}, {self.primitive_name}: "
            f"args: {args}, kws: {kws}, "
        )
        print(msg)
        result = super().apply(args, kws)
        print(msg + f"result: {result}")
        return result

    def _select(self, cases, args, kws):
        msg = (
            f"{self.__class__.__name__}._select() for "
            f"{self.key}, {self.primitive_name}: "
            f"cases: {cases}, args: {args}, kws: {kws}, "
        )
        result = super()._select(cases, args, kws)
        print(msg + f"result: {result}")
        return result

    def _validate_src_dst(self, src, dst):
        """
        Validate that *src* and *dst* are both provided, are device arrays,
        and have compatible types (dtype, ndim, layout).  Raise TypingError
        if any of the checks fail.  Return None if all checks pass.
        """
        if src is None or dst is None:
            raise errors.TypingError(
                f"{self.primitive_name} needs both 'src' and 'dst' arrays"
            )

        invalid_types = not isinstance(src, types.Array) or not isinstance(
            dst, types.Array
        )
        if invalid_types:
            raise errors.TypingError(
                f"{self.primitive_name} requires both 'src' and 'dst' to be "
                "device arrays"
            )

        # Mismatched types.
        if src.dtype != dst.dtype:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' and 'dst' to have the "
                f"same dtype (got {src.dtype} vs {dst.dtype})"
            )

        # Mismatched dimensions.
        if src.ndim != dst.ndim:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' and 'dst' to have the "
                f"same number of dimensions (got {src.ndim} vs {dst.ndim})"
            )

        # Mismatched layout if neither is 'A'.
        invalid_layout = (
            src.layout != "A" and dst.layout != "A" and src.layout != dst.layout
        )
        if invalid_layout:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' and 'dst' to have the "
                f"same layout (got {src.layout!r} vs {dst.layout!r})"
            )

    def _validate_positive_integer_literal(self, value, param_name):
        """
        Validate that *value* is a positive integer literal and return it as
        an IntegerLiteral type.  If the underlying literal value is less than
        or equal to zero, raise a TypingError.  Otherwise, return None,
        indicating that this type is not supported.
        """
        if isinstance(value, types.IntegerLiteral):
            if value.literal_value <= 0:
                raise errors.TypingError(
                    f"'{param_name}' must be a positive integer; "
                    f"got {value.literal_value}"
                )
            return value
        return None

    def _validate_threads_per_block(self, threads_per_block):
        return self._validate_positive_integer_literal(
            threads_per_block,
            "threads_per_block",
        )

    def _validate_items_per_thread(self, items_per_thread):
        return self._validate_positive_integer_literal(
            items_per_thread,
            "items_per_thread",
        )

    def _validate_algorithm(self, algorithm):
        if algorithm is None:
            return

        enum_cls = self.algorithm_enum
        enum_name = enum_cls.__name__
        user_facing_name = f"cuda.{enum_name}"

        if not isinstance(algorithm, types.EnumMember):
            msg = (
                f"algorithm for {self.primitive_name} must be a member "
                f"of {user_facing_name}, got {algorithm}"
            )
            raise errors.TypingError(msg)

        if algorithm.instance_class is not enum_cls:
            name = algorithm.instance_class.__name__
            msg = (
                f"algorithm for {self.primitive_name} must be a member "
                f"of {user_facing_name}, got {name} "
            )
            raise errors.TypingError(msg)

    def _validate_args_and_create_signature(
        self, src, dst, threads_per_block, items_per_thread, algorithm=None
    ):
        """
        Validate the arguments passed to the cooperative load/store function.
        This includes checking that src and dst are valid arrays, and that
        threads_per_block and items_per_thread are positive integer literals,
        and algorithm, if provided, is suitable for the given primitive.
        """
        self._validate_src_dst(src, dst)

        if not self._validate_threads_per_block(threads_per_block):
            return

        if not self._validate_items_per_thread(items_per_thread):
            return

        self._validate_algorithm(algorithm)

        # If we reach here, all arguments are valid.
        if self.src_first:
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

        sig = signature(
            types.void,
            *arglist,
        )

        if algorithm is None:
            if not hasattr(self.context, "_coop"):
                self.context._coop = defaultdict(dict)
            self.context._coop[sig]["algorithm"] = self.default_algorithm.value

        return sig


class LoadMixin:
    src_first = True

    def generic(self):
        def typer(
            src,
            dst,
            threads_per_block,
            items_per_thread,
            algorithm=None,
        ):
            return self._validate_args_and_create_signature(
                src,
                dst,
                threads_per_block,
                items_per_thread,
                algorithm,
            )

        return typer


class StoreMixin:
    src_first = False

    def generic(self):
        def typer(
            dst,
            src,
            threads_per_block,
            items_per_thread,
            algorithm=None,
        ):
            return self._validate_args_and_create_signature(
                src,
                dst,
                threads_per_block,
                items_per_thread,
                algorithm,
            )

        return typer


@register
class Coop_block_load(Coop_load_store_decl, LoadMixin):
    key = cuda.block.load
    primitive_name = "cuda.block.load"
    algorithm_enum = BlockLoadAlgorithm
    default_algorithm = BlockLoadAlgorithm.DIRECT


@register
class Coop_warp_load(Coop_load_store_decl, LoadMixin):
    key = cuda.warp.load
    primitive_name = "cuda.warp.load"
    algorithm_enum = WarpLoadAlgorithm
    default_algorithm = WarpLoadAlgorithm.DIRECT


@register
class Coop_block_store(Coop_load_store_decl, StoreMixin):
    key = cuda.block.store
    primitive_name = "cuda.block.store"
    algorithm_enum = BlockStoreAlgorithm
    default_algorithm = BlockStoreAlgorithm.DIRECT


@register
class Coop_warp_store(Coop_load_store_decl, StoreMixin):
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
