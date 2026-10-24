import json

import pytest

from telecom_rag.tools import PlanBook


@pytest.fixture(scope="module")
def plans():
    return PlanBook()


@pytest.mark.parametrize("text, plan_id", [
    ("5G畅享59元套餐", "5g_59"),
    ("畅享59", "5g_59"),
    ("我是159元套餐", "5g_159"),       # 不能误匹配成 59 元
    ("199元套餐", "family_199"),       # 不能误匹配成 19 元
    ("校园卡", "campus_29"),
    ("守护卡", "guard_19"),
    ("39", "5g_39"),
])
def test_resolve_plan(plans, text, plan_id):
    assert plans.resolve(text)["id"] == plan_id


def test_resolve_unknown(plans):
    assert plans.resolve("129元") is None
    assert plans.resolve("物联网卡") is None


@pytest.mark.parametrize("plan, data, voice, cards, total", [
    ("5G畅享39元", 70, 100, 0, 189),      # 套外 40GB×5=200，封顶 150
    ("5G畅享59元", 80, 350, 0, 126.5),    # 20GB×3 + 50分钟×0.15
    ("5G畅享99元", 120.5, 600, 2, 192),   # 20.5GB 向上取整为 21GB
    ("4G守护卡19元", 12, 150, 0, 36.5),
    ("5G尊享159元", 230, 1200, 3, 279),   # 2 张免费副卡，第 3 张收 10 元
    ("5G畅享59元", 65.5, 320, 0, 80),     # 与 02_overage_billing.md 的计算示例一致
    ("5G畅享59元", 75, 300, 0, 104),
    ("5G畅享59元", 30, 0, 0, 59),         # 没用超只收月费
])
def test_calculate_fee(plans, plan, data, voice, cards, total):
    result = plans.calculate_fee(plan, data, voice, cards)
    assert result["合计(元)"] == pytest.approx(total)


def test_calculate_fee_accepts_strings(plans):
    assert plans.calculate_fee("59元套餐", "80GB", "350分钟")["合计(元)"] == pytest.approx(126.5)


def test_calculate_fee_errors(plans):
    assert "error" in plans.calculate_fee("不存在的套餐", 10)
    assert "error" in plans.calculate_fee("5G畅享39元", 10, 0, 1)  # 39 元套餐不能加副卡


def test_recommend_plan(plans):
    result = plans.recommend_plan(45, 200)
    assert result["最省钱的套餐"] == "5G畅享59元"
    fees = [r["预计月费用(元)"] for r in result["费用排名"]]
    assert fees == sorted(fees)
    assert all(r["套餐"] != "青春校园卡29元" for r in result["费用排名"])  # 非学生不推荐校园卡


def test_recommend_plan_student(plans):
    assert plans.recommend_plan(25, 150, is_student=True)["最省钱的套餐"] == "青春校园卡29元"


def test_query_plans(plans):
    names = [r["套餐"] for r in plans.query_plans(max_monthly_fee=60, need_5g=True)["套餐列表"]]
    assert names == ["5G畅享39元", "5G畅享59元", "青春校园卡29元"]
    assert [r["套餐"] for r in plans.query_plans(keyword="校园")["套餐列表"]] == ["青春校园卡29元"]
    assert len(plans.query_plans(min_data_gb=100)["套餐列表"]) == 3


def test_run_returns_json(plans):
    data = json.loads(plans.run("calculate_fee", {"plan": "5G畅享39元", "data_gb": 70}))
    assert data["合计(元)"] == 189
    assert "error" in json.loads(plans.run("unknown_tool", {}))
    assert "error" in json.loads(plans.run("calculate_fee", {"plan": "5G畅享39元", "foo": 1}))


@pytest.mark.parametrize("field,value", [
    ("data_gb", -10), ("data_gb", "not-a-number"), ("data_gb", "80MB"),
    ("data_gb", None), ("data_gb", ""), ("data_gb", True),
    ("data_gb", float("inf")), ("data_gb", float("nan")), ("data_gb", "1e309"),
    ("data_gb", 10**1000), ("voice_minutes", -1), ("voice_minutes", "30GB"),
    ("secondary_cards", 1.9), ("secondary_cards", "1.9张"), ("secondary_cards", -1),
])
def test_invalid_usage_returns_an_error_instead_of_a_bill(plans, field, value):
    args = {"plan": "159元", "data_gb": 10, field: value}
    result = json.loads(plans.run("calculate_fee", args))
    assert "error" in result and "合计(元)" not in result


@pytest.mark.parametrize("name,args", [
    ("recommend_plan", {"data_gb": 25, "is_student": "false"}),
    ("recommend_plan", {"data_gb": 25, "is_student": 1}),
    ("recommend_plan", {"data_gb": -1}),
    ("recommend_plan", {"data_gb": 25, "secondary_cards": 1.9}),
    ("recommend_plan", {"data_gb": 25, "secondary_cards": 999}),
    ("query_plans", {"need_5g": "false"}),
    ("query_plans", {"max_monthly_fee": "unknown"}),
    ("query_plans", {"min_data_gb": -1}),
    ("calculate_fee", {"plan": "59元"}),
    ("calculate_fee", []),
])
def test_invalid_parameters_are_rejected_by_every_tool(plans, name, args):
    assert "error" in json.loads(plans.run(name, args))


def test_explicit_false_and_zero_remain_valid(plans):
    result=json.loads(plans.run("recommend_plan", {"data_gb": 25, "is_student": False}))
    assert all(r["套餐"] != "青春校园卡29元" for r in result["费用排名"])
    assert plans.calculate_fee("159元", "0GB", "0分钟", "1张副卡")["副卡数"] == 1
