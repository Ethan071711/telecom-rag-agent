from telecom_rag.textutil import REFUSAL_TEXT, extract_citations, is_refusal, normalize, numeric_claims


def test_refusal_detection():
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal("抱歉，知识库中没有找到相关信息。")
    assert is_refusal("资料中没有提到卫星电话的资费，建议联系人工客服。")
    assert not is_refusal("停机保号每月 5 元，最长 12 个月 [1]。")


def test_extract_citations():
    assert extract_citations("A[1]。B[2][1]。C【3】，D[4, 5]") == [1, 2, 3, 4, 5]
    assert extract_citations("没有引用") == []


def test_numeric_claims_and_normalize():
    assert numeric_claims("合计 189 元，套外 40GB [2]，降速至 1Mbps") == ["189", "40", "1"]
    assert normalize("３０ GB") == "30g"
