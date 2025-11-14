from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation, getcontext
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from loom.interfaces.tool import BaseTool


def _to_decimal(val: Any) -> Decimal:
    if isinstance(val, Decimal):
        return val
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("数值格式错误") from exc


def _round_decimal(x: Decimal, places: int) -> Decimal:
    quant = Decimal(1).scaleb(-places)  # 10^-places
    return x.quantize(quant, rounding=ROUND_HALF_UP) if places >= 0 else x


def _format_percentage(ratio: Decimal, perc_places: int) -> str:
    pct = ratio * Decimal(100)
    pct = _round_decimal(pct, perc_places)
    # 标准化避免科学计数法
    return f"{pct:.{perc_places}f}%"


def _normalize_mode(mode: Optional[str]) -> Tuple[str, Optional[str]]:
    if not mode:
        return "product", None
    m = mode.strip().lower()
    if m in ("product",):
        return "product", None
    if m in ("demo",):
        return "demo", None
    if m in ("r&d", "dev", "rnd", "r-and-d"):
        return "r&d", None
    return "product", "未知模式，已回退为 product"


def _dec_context() -> None:
    # 设定较高精度，避免中间计算丢失；最终由量化控制输出精度
    getcontext().prec = 28


def _single_compute(
    *,
    cost: Any,
    revenue: Any,
    mode: Optional[str] = None,
    output: Optional[Literal["structured", "text"]] = None,
    precision_ratio: int = 4,
    precision_percentage: int = 2,
) -> Dict[str, Any] | str:
    _dec_context()

    norm_mode, warn = _normalize_mode(mode)

    c = _to_decimal(cost)
    r = _to_decimal(revenue)

    if c <= 0:
        raise ValueError("成本必须大于 0")

    ratio_raw = (r - c) / c
    ratio = _round_decimal(ratio_raw, precision_ratio)
    percentage = _format_percentage(ratio_raw, precision_percentage)

    explanation = (
        f"投资回报率 {percentage}（成本 {c}、收益 {r}）。"
        if ratio_raw >= 0
        else f"投资回报率 {percentage}（亏损，成本 {c}、收益 {r}）。"
    )

    # 决定输出形式
    if output:
        out_mode = output
    else:
        out_mode = "structured" if norm_mode == "product" else "text"

    warnings: List[str] = []
    if warn:
        warnings.append(warn)

    if out_mode == "structured":
        result: Dict[str, Any] = {
            "ratio": ratio,
            "percentage": percentage,
            "explanation": explanation,
        }
        if warnings:
            result["warnings"] = warnings
        return result
    else:
        warn_txt = (" （" + "; ".join(warnings) + ")") if warnings else ""
        return f"{explanation}{warn_txt}"


a_Output = Literal["structured", "text"]


class RoiArgs(BaseModel):
    cost: float | int | str = Field(description="成本，必须大于 0")
    revenue: float | int | str = Field(description="收益")
    mode: Optional[str] = Field(default=None, description="product|demo|r&d")
    output: Optional[a_Output] = Field(default=None, description="structured|text")
    precision_ratio: int = Field(default=4, description="比率小数位")
    precision_percentage: int = Field(default=2, description="百分比小数位")


class RoiTool(BaseTool):
    name = "roi"
    description = "计算单笔 ROI（比率与百分比），支持 product/demo/r&d 模式"
    args_schema = RoiArgs

    # 纯计算、只读
    is_read_only = True
    category = "general"

    async def run(self, **kwargs) -> Any:
        args = self.args_schema(**kwargs)  # type: ignore
        return _single_compute(
            cost=args.cost,
            revenue=args.revenue,
            mode=args.mode,
            output=args.output,
            precision_ratio=args.precision_ratio,
            precision_percentage=args.precision_percentage,
        )


class BatchArgs(BaseModel):
    items: List[Dict[str, Any]] = Field(description="批量条目列表")
    mode: Optional[str] = Field(default=None, description="顶层模式，items 中缺省时继承")
    output: Optional[a_Output] = Field(default=None, description="顶层输出，items 中缺省时继承")
    precision_ratio: int = Field(default=4)
    precision_percentage: int = Field(default=2)


def _aggregate(valid_ratios: Sequence[Decimal]) -> Dict[str, Optional[Decimal]]:
    if not valid_ratios:
        return {"avg_ratio": None, "min_ratio": None, "max_ratio": None}
    s = sum(valid_ratios, Decimal(0))
    n = Decimal(len(valid_ratios))
    return {
        "avg_ratio": s / n,
        "min_ratio": min(valid_ratios),
        "max_ratio": max(valid_ratios),
    }


class RoiBatchTool(BaseTool):
    name = "roi_batch"
    description = "批量计算 ROI，返回逐条结果与汇总统计"
    args_schema = BatchArgs

    is_read_only = True
    category = "general"

    async def run(self, **kwargs) -> Any:
        args = self.args_schema(**kwargs)  # type: ignore

        _dec_context()
        norm_mode, warn_top = _normalize_mode(args.mode)
        inherited_output = args.output

        results: List[Any] = []
        valid_ratios: List[Decimal] = []
        warnings_top: List[str] = []
        if warn_top:
            warnings_top.append(warn_top)

        for idx, it in enumerate(args.items or []):
            try:
                it_mode = it.get("mode", norm_mode)
                it_out = it.get("output", inherited_output)
                res = _single_compute(
                    cost=it.get("cost"),
                    revenue=it.get("revenue"),
                    mode=it_mode,
                    output=it_out,
                    precision_ratio=args.precision_ratio,
                    precision_percentage=args.precision_percentage,
                )
                results.append(res)
                if isinstance(res, dict) and "ratio" in res:
                    valid_ratios.append(_to_decimal(res["ratio"]))
                else:
                    c = _to_decimal(it.get("cost"))
                    r = _to_decimal(it.get("revenue"))
                    if c > 0:
                        valid_ratios.append((r - c) / c)
            except (ValueError, InvalidOperation, TypeError) as e:
                results.append({"index": idx, "message": str(e)})

        summary = _aggregate(valid_ratios)

        out_mode = inherited_output or ("structured" if norm_mode == "product" else "text")
        if out_mode == "structured":
            payload: Dict[str, Any] = {
                "results": results,
                "summary": summary,
                "errors_count": sum(
                    1 for r in results if isinstance(r, dict) and "message" in r and "ratio" not in r
                ),
            }
            if warnings_top:
                payload["warnings"] = warnings_top
            return payload
        else:
            lines: List[str] = []
            if warnings_top:
                lines.append("警告: " + "; ".join(warnings_top))
            for i, r in enumerate(results):
                if isinstance(r, dict) and "message" in r and "ratio" not in r:
                    lines.append(f"[{i}] 错误: {r['message']}")
                else:
                    lines.append(f"[{i}] {r}")
            if summary["avg_ratio"] is not None:
                lines.append(
                    "汇总: 平均=%.4f, 最小=%.4f, 最大=%.4f"
                    % (
                        summary["avg_ratio"],
                        summary["min_ratio"],
                        summary["max_ratio"],
                    )
                )
            else:
                lines.append("汇总: 无有效项")
            return "\n".join(lines)
