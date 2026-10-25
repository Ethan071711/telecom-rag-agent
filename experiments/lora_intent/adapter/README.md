---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: peft
---
# Small telecom intent adapter

LoRA q_proj/v_proj adapter trained locally on 100 project-authored synthetic utterances. See the parent experiment README and the full measured results for the fixed 25 validation / 50 test examples and limitations. The adapter is a new 2026-10-25 learning experiment, not a production or internship model.

Base model: Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca. Base and derivative adapter use Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0 . Original model card: https://huggingface.co/Qwen/Qwen3-0.6B .
