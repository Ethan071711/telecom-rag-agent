import pytest

from experiments.lora_intent.train import LABELS,encode_example,load_data,score


def test_fixed_intent_splits_are_disjoint_and_cover_labels():
    _,splits=load_data()
    assert {k:len(v) for k,v in splits.items()}=={'train':100,'validation':25,'test':50}
    assert all({row['label'] for row in rows}==set(LABELS) for rows in splits.values())


def test_prompt_tokens_are_excluded_from_training_loss():
    class Tokenizer:
        eos_token='END'
        def apply_chat_template(self,*args,**kwargs):return {'input_ids':[11,12,13]}
        def __call__(self,*args,**kwargs):return {'input_ids':[21,22]}
    encoded=encode_example(Tokenizer(),'套餐多少钱？','plans')
    assert encoded=={'input_ids':[11,12,13,21,22],'labels':[-100,-100,-100,21,22]}


def test_invalid_model_output_is_counted_as_incorrect():
    rows=[{'label':label,'prediction':label} for label in LABELS]
    rows.append({'label':'fee','prediction':'I think the answer is fee'})
    metrics=score(rows)
    assert metrics['cases']==6 and metrics['correct']==5 and metrics['invalid_outputs']==1
    assert metrics['confusion_matrix']['fee']['invalid']==1
    assert metrics['accuracy']==pytest.approx(5/6)
