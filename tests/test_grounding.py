from telecom_rag.grounding import issues, quantities, tool_facts, extractive_fallback, fee_answer
from telecom_rag.tools import PlanBook


def test_unit_and_cited_source_are_both_required():
    evidence = ['充值一般实时到账，最迟2小时。充值券30元。', '复机可能延迟30分钟。']
    assert issues('充值超过30分钟请联系客服 [1]。', evidence, [])
    assert not issues('充值最迟2小时到账 [1]。', evidence, [])
    assert issues('充值最迟2小时到账 [9]。', evidence, [])
    assert issues('充值最迟2小时到账。', evidence, [])


def test_structured_tool_numbers_keep_their_units():
    result = '{"套餐列表":[{"月费(元)":39,"国内通用流量(GB)":30}]}'
    calls = [{'name': 'query_plans', 'result': result}]
    assert not issues('月费39元，包含30GB。', [], calls)
    assert issues('月费30元，包含39GB。', [], calls)
    assert tool_facts('{"error":"数值30不能使用"}') == ''


def test_numeric_normalization_does_not_depend_on_formatting():
    assert quantities('３０.０ GB，５块【1】') == quantities('30GB，5元')
    assert issues('资料中没有明确说明，但应该30分钟到账。', ['最迟2小时'], [])


def test_fallback_quotes_expanded_source_with_correct_section_ids():
    evidence=['余额不能提现', '一般实时到账；繁忙时最迟2小时，超过后保存支付订单号联系客服。']
    answer=extractive_fallback('30分钟到账 [1]',evidence,[],[1,2])
    assert evidence[1]+' [2]' in answer
    assert '30分钟' not in answer
    assert not issues(answer,evidence,[])


def test_fee_answer_preserves_free_and_paid_card_conditions():
    book=PlanBook()
    result=book.run('calculate_fee',{'plan':'5G尊享159元','data_gb':230,'voice_minutes':1200,'secondary_cards':3})
    calls=[{'name':'calculate_fee','result':result}]
    answer=fee_answer(calls)
    assert '279' in answer and '共3张，其中免费2张、收费1张' in answer
    assert not issues(answer,[],calls)
    assert fee_answer([{'name':'calculate_fee','result':'{"error":"套餐不存在"}'}]) is None
