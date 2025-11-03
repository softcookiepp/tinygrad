from tinygrad.dtype import DType, PtrDType, dtypes, AddrSpace, truncate
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, GroupOp
from tinygrad.renderer.cstyle import OpenCLRenderer, base_rewrite, extra_pm, wmma_args
from tinygrad.helpers import strip_parens

class ClspvRenderer(OpenCLRenderer):
  device = "VK"
  string_rewrite = PatternMatcher([
  (UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx.nan)})" if math.isnan(x.arg) else None),
  (UPat(Ops.CONST, dtype=dtypes.half, name="x"), lambda ctx,x: f"{x.arg}".replace("inf", "INFINITY") ),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{x.arg}".replace("inf", "INFINITY") ),
  (UPat(Ops.CONST, dtype=dtypes.double, name="x"), lambda ctx,x: f"{x.arg}".replace("inf", "INFINITY") ),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: f"{int(x.arg)}"),
  (UPat(Ops.CONST, dtype=dtypes.char, name="x"), lambda ctx,x: f"{x.arg}"),
  (UPat(Ops.CONST, dtype=dtypes.short, name="x"), lambda ctx,x: f"{x.arg}"),
  (UPat(Ops.CONST, dtype=dtypes.int, name="x"), lambda ctx,x: f"{x.arg}"),
  (UPat(Ops.CONST, dtype=dtypes.long, name="x"), lambda ctx,x: f"{x.arg}"),
  (UPat(Ops.CONST, dtype=dtypes.uchar, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}"),
  (UPat(Ops.CONST, dtype=dtypes.ushort, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}"),
  (UPat(Ops.CONST, dtype=dtypes.uint, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}"),
  (UPat(Ops.CONST, dtype=dtypes.ulong, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}"),
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, ctx.infinity)})"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, f'-{ctx.infinity}')})")
  
  ]) + OpenCLRenderer.string_rewrite
  
  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    out = super().render_kernel(function_name, kernel, bufs, uops, prefix)
    if False:
      self._path = "rendered"
      os.makedirs(self._path, exist_ok = True)
      with open(os.path.join(self._path, f"{function_name}.cl"), "w") as f:
        f.write(out)
    return out
