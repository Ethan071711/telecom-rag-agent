"""Small, reproducible LoRA intent experiment; the main RAG application is unchanged."""
import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

ROOT=Path(__file__).resolve().parent
MODEL='Qwen/Qwen3-0.6B'
REVISION='c1899de289a04d12100db370d81485cdf75e47ca'
LABELS=('knowledge','plans','fee','recommend','unsupported')
CONFIG={'seed':42,'epochs':6,'batch_size':4,'gradient_accumulation':4,'learning_rate':0.0002,
        'max_length':384,'lora_r':8,'lora_alpha':16,'lora_dropout':0.05,'target_modules':['q_proj','v_proj']}
PROMPT=('将用户的中文请求分类，只输出一个英文标签，不解释。\n'
        'knowledge：通信业务规则、办理方法、故障处理。\n'
        'plans：查询在售手机套餐的配置、资费或筛选列表。\n'
        'fee：用户已选定套餐，计算其实际用量对应的账单。\n'
        'recommend：用户未选定套餐，按需求比较并推荐合适的手机套餐。\n'
        'unsupported：不在本客服范围的请求，含企业专线、物联网卡、卫星电话、座机及云盘业务。')


def load_data(path=ROOT/'data.json'):
    source=json.loads(path.read_text('utf-8'))
    splits={split:[{'text':text,'label':label} for label,items in source[split].items() for text in items]
            for split in ('train','validation','test')}
    seen=set()
    for split,items in splits.items():
        if {item['label'] for item in items}!=set(LABELS):
            raise ValueError('Every split must cover every intent')
        for item in items:
            key=''.join(item['text'].split())
            if key in seen:
                raise ValueError('Duplicate utterance across or inside data splits')
            seen.add(key)
    return source['provenance'],splits


def encode_example(tokenizer,text,label):
    prompt=tokenizer.apply_chat_template([{'role':'system','content':PROMPT},{'role':'user','content':text}],
        tokenize=True,return_dict=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
    target=tokenizer(label+tokenizer.eos_token,add_special_tokens=False)['input_ids']
    if len(prompt)+len(target)>CONFIG['max_length']:
        raise ValueError('Example exceeds configured length; do not truncate away the target')
    return {'input_ids':prompt+target,'labels':[-100]*len(prompt)+target}


def score(rows):
    confusion={label:{pred:0 for pred in (*LABELS,'invalid')} for label in LABELS}
    for row in rows:
        prediction=row['prediction'] if row['prediction'] in LABELS else 'invalid'
        confusion[row['label']][prediction]+=1
    f1=[]
    for label in LABELS:
        tp=confusion[label][label]
        fp=sum(confusion[other][label] for other in LABELS if other!=label)
        fn=sum(confusion[label][pred] for pred in (*LABELS,'invalid') if pred!=label)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'cases':len(rows),'correct':sum(row['prediction']==row['label'] for row in rows),
            'accuracy':sum(row['prediction']==row['label'] for row in rows)/len(rows),
            'macro_f1':sum(f1)/len(f1),'invalid_outputs':sum(row['prediction'] not in LABELS for row in rows),
            'confusion_matrix':confusion}


def evaluate(model,tokenizer,items):
    import torch
    rows=[]
    model.eval()
    for item in items:
        encoded=encode_example(tokenizer,item['text'],item['label'])
        prompt_length=encoded['labels'].count(-100)
        inputs=torch.tensor([encoded['input_ids'][:prompt_length]],device=model.device)
        with torch.inference_mode():
            output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),do_sample=False,
                max_new_tokens=12,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        prediction=tokenizer.decode(output[0,prompt_length:],skip_special_tokens=True).strip()
        rows.append({**item,'prediction':prediction})
    return {'metrics':score(rows),'details':rows}


def main(output):
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,PeftModel,TaskType,get_peft_model
    provenance,splits=load_data()
    random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed']);torch.set_num_threads(4)
    output.mkdir(parents=True,exist_ok=True)
    manifest={'config':CONFIG,'base_model':MODEL,'base_revision':REVISION,'prompt':PROMPT,
        'dataset_sha256':hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'split_counts':{split:dict(Counter(row['label'] for row in rows)) for split,rows in splits.items()},
        'frozen_before_evaluation_at_utc':datetime.now(timezone.utc).isoformat()}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    tokenizer.pad_token=tokenizer.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=dtype,
        device_map={'':0} if torch.cuda.is_available() else 'cpu',attn_implementation='sdpa')
    print('BASELINE: validation and test, same frozen prompt',flush=True)
    baseline={split:evaluate(model,tokenizer,splits[split]) for split in ('validation','test')}
    print(json.dumps({split:d['metrics'] for split,d in baseline.items()},ensure_ascii=False),flush=True)
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],lora_dropout=CONFIG['lora_dropout'],target_modules=CONFIG['target_modules']))
    trainable,total=model.get_nb_trainable_parameters()
    assert trainable>0 and all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
    encoded=[encode_example(tokenizer,row['text'],row['label']) for row in splits['train']]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['learning_rate'])
    model.config.use_cache=False
    epoch_losses=[];steps=0
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for epoch in range(CONFIG['epochs']):
        model.train();order=list(range(len(encoded)));random.shuffle(order)
        batches=[order[i:i+CONFIG['batch_size']] for i in range(0,len(order),CONFIG['batch_size'])]
        losses=[]
        for start in range(0,len(batches),CONFIG['gradient_accumulation']):
            window=batches[start:start+CONFIG['gradient_accumulation']]
            optimizer.zero_grad(set_to_none=True)
            for indexes in window:
                samples=[encoded[i] for i in indexes];length=max(len(s['input_ids']) for s in samples)
                ids=torch.tensor([s['input_ids']+[tokenizer.pad_token_id]*(length-len(s['input_ids'])) for s in samples],device=model.device)
                labels=torch.tensor([s['labels']+[-100]*(length-len(s['labels'])) for s in samples],device=model.device)
                mask=torch.tensor([[1]*len(s['input_ids'])+[0]*(length-len(s['input_ids'])) for s in samples],device=model.device)
                loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                losses.append(float(loss.detach()))
                (loss/len(window)).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step();steps+=1
        average=sum(losses)/len(losses);epoch_losses.append(average)
        print(f'EPOCH {epoch+1}/{CONFIG["epochs"]}: loss={average:.6f}, optimizer_steps={steps}',flush=True)
    training_seconds=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    model.config.use_cache=True
    adapter=output/'adapter';model.save_pretrained(adapter,safe_serialization=True)
    # Reload only the saved adapter into the same frozen base weights before evaluating.
    base=model.unload();del model,optimizer
    model=PeftModel.from_pretrained(base,adapter)
    tuned={split:evaluate(model,tokenizer,splits[split]) for split in ('validation','test')}
    report={'measured_at_utc':datetime.now(timezone.utc).isoformat(),'manifest':manifest,'data_provenance':provenance,
        'method':'LoRA supervised causal-LM fine-tuning; prompt/padding tokens masked; no quantization, no QLoRA claim',
        'trainable_parameters':trainable,'total_parameters':total,'trainable_fraction':trainable/total,
        'optimizer_steps':steps,'epoch_train_loss':epoch_losses,'training_seconds':training_seconds,
        'gpu_peak_allocated_bytes':peak,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'packages':{p:version(p) for p in ('torch','transformers','peft','accelerate')},
        'baseline':baseline,'reloaded_adapter':tuned,'test_accuracy_change':tuned['test']['metrics']['accuracy']-baseline['test']['metrics']['accuracy'],
        'adapter_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'paid_api_calls':0,'limitations':['Small balanced synthetic utterances from the same authoring process, not a real-user generalization benchmark.',
        'Fixed six epochs; no test-based checkpoint or hyperparameter selection. No production routing integration.',
        'This is a new local experiment, not a historical internship result or QLoRA experiment.']}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'baseline_test':baseline['test']['metrics'],'adapter_test':tuned['test']['metrics'],
        'training_seconds':training_seconds,'peak_allocated_bytes':peak,'trainable_parameters':trainable},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parents[1]/'storage/lora-intent-20261025')
    main(parser.parse_args().output)
