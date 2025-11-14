import pytest
from decimal import Decimal

from loom.builtin.tools.roi import RoiTool, RoiBatchTool


@pytest.mark.asyncio
async def test_roi_single_positive_structured():
    tool = RoiTool()
    res = await tool.run(cost=100, revenue=130)
    assert isinstance(res, dict)
    assert res["ratio"] == Decimal("0.3000")
    assert res["percentage"] == "30.00%"
    assert "投资回报率" in res["explanation"]


@pytest.mark.asyncio
async def test_roi_single_negative_structured():
    tool = RoiTool()
    res = await tool.run(cost=200, revenue=180)
    assert isinstance(res, dict)
    assert res["ratio"] == Decimal("-0.1000")
    assert res["percentage"] == "-10.00%"
    assert "亏损" in res["explanation"]


@pytest.mark.asyncio
async def test_roi_single_cost_zero_error():
    tool = RoiTool()
    with pytest.raises(ValueError):
        await tool.run(cost=0, revenue=100)


@pytest.mark.asyncio
async def test_roi_demo_mode_text_output():
    tool = RoiTool()
    res = await tool.run(cost=50, revenue=75, mode="demo")
    assert isinstance(res, str)
    assert "50" in res and "75" in res
    assert "%" in res


@pytest.mark.asyncio
async def test_roi_illegal_mode_falls_back_with_warning():
    tool = RoiTool()
    res = await tool.run(cost=100, revenue=120, mode="unknown-mode")
    assert isinstance(res, dict)
    assert "warnings" in res and any("回退" in w for w in res["warnings"])  # 非致命警告


@pytest.mark.asyncio
async def test_roi_batch_mixed_items_structured():
    tool = RoiBatchTool()
    payload = {
        "items": [
            {"cost": 100, "revenue": 130},
            {"cost": 200, "revenue": 180},
            {"cost": 0, "revenue": 99},  # 错误项
        ]
    }
    res = await tool.run(**payload)
    assert isinstance(res, dict)
    assert "results" in res and "summary" in res
    # 三条结果：两条成功、一条错误
    assert len(res["results"]) == 3
    assert res["errors_count"] == 1
    # 汇总基于两条有效项
    summ = res["summary"]
    assert summ["avg_ratio"] is not None
    assert summ["min_ratio"] is not None
    assert summ["max_ratio"] is not None
