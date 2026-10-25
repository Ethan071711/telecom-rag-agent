"""Run the saved intent adapter locally; no business tool is executed."""
import argparse
import json
from pathlib import Path

from train import LABELS,MODEL,PROMPT,REVISION


def main(text,adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    tokenizer.pad_token=tokenizer.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=dtype,
        device_map={'':0} if torch.cuda.is_available() else 'cpu')
    model=PeftModel.from_pretrained(base,adapter).eval()
    inputs=tokenizer.apply_chat_template([{'role':'system','content':PROMPT},{'role':'user','content':text}],
        tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False).to(model.device)
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,max_new_tokens=12,
            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    raw=tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    print(json.dumps({'text':text,'intent':raw if raw in LABELS else None,'raw_output':raw,
                     'scope':'Experimental classifier only; no tool execution'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    parser.add_argument('--adapter',type=Path,default=Path(__file__).resolve().parent/'adapter')
    args=parser.parse_args();main(args.text,args.adapter)
