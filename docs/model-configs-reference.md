# KV Cache Calculator - Model Configuration Reference

> This document records all model configurations in `data/kv_cache_calculator/models.yaml`.
> Parameters are sourced from official HuggingFace config.json files.

## Formulas

| Formula | Description | Key Fields |
|---------|-------------|------------|
| `standard_gqa` | Standard Grouped Query Attention | num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim |
| `mla` | Multi-head Latent Attention (DeepSeek) | num_hidden_layers, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, num_key_value_heads |
| `dsa_mla` | DSA + MLA with indexer | Same as mla + index_head_dim, index_n_heads, index_topk |
| `deepseek_v4_hybrid` | DeepSeek V4 hybrid (sliding window + compression) | num_hidden_layers, head_dim, sliding_window, compress_ratios, index_head_dim, index_topk |

---

## Pre-existing Models (before PR #7)

### DeepSeek V4 (deepseek_v4_hybrid)

| Model | Layers | head_dim | sliding_window | index_head_dim | index_topk | KV Heads | Max Ctx |
|-------|--------|----------|---------------|----------------|-----------|----------|---------|
| DeepSeek V4 Pro | 61 | 512 | 128 | 128 | 1024 | 1 | 1048576 |
| DeepSeek V4 Flash | 43 | 512 | 128 | 128 | 512 | 1 | 1048576 |

Note: Uses compress_ratios array per-layer. See YAML for full arrays.

Source: `https://huggingface.co/deepseek-ai/DeepSeek-V4-*/raw/main/config.json`

### DeepSeek V3.2 (dsa_mla)

| Model | Layers | kv_lora_rank | qk_nope_head_dim | qk_rope_head_dim | qk_head_dim | v_head_dim | KV Heads | index_head_dim | index_n_heads | index_topk | Max Ctx |
|-------|--------|-------------|------------------|------------------|------------|-----------|----------|----------------|--------------|-----------|---------|
| DeepSeek V3.2 | 61 | 512 | 128 | 64 | 192 | 128 | 128 | 128 | 64 | 2048 | 163840 |

Source: `https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json`

### GLM-5 / 5.1 (dsa_mla)

| Model | Layers | kv_lora_rank | qk_nope_head_dim | qk_rope_head_dim | qk_head_dim | v_head_dim | KV Heads | index_head_dim | index_n_heads | index_topk | Max Ctx |
|-------|--------|-------------|------------------|------------------|------------|-----------|----------|----------------|--------------|-----------|---------|
| GLM-5 | 78 | 512 | 192 | 64 | 256 | 256 | 64 | 128 | 32 | 2048 | 202752 |
| GLM-5.1 | 78 | 512 | 192 | 64 | 256 | 256 | 64 | 128 | 32 | 2048 | 202752 |

Source: `https://huggingface.co/zai-org/GLM-5*/raw/main/config.json`

### Kimi K2.5 / K2.6 (MLA)

| Model | Layers | kv_lora_rank | qk_rope_head_dim | qk_nope_head_dim | v_head_dim | KV Heads | Max Ctx |
|-------|--------|-------------|------------------|------------------|-----------|----------|---------|
| Kimi K2.5 | 61 | 512 | 64 | 128 | 128 | 64 | 262144 |
| Kimi K2.6 | 61 | 512 | 64 | 128 | 128 | 64 | 262144 |

Source: `https://huggingface.co/moonshotai/Kimi-K2.*/raw/main/config.json`

### Qwen3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Qwen3-235B-A22B | 94 | 64 | 4 | 128 | 40960 |
| Qwen3-32B | 64 | 64 | 8 | 128 | 40960 |
| Qwen3-30B-A3B | 48 | 32 | 4 | 128 | 40960 |
| Qwen3-14B | 40 | 40 | 8 | 128 | 40960 |
| Qwen3-8B | 36 | 32 | 8 | 128 | 40960 |
| Qwen3-4B | 36 | 32 | 8 | 128 | 40960 |
| Qwen3-1.7B | 28 | 16 | 8 | 128 | 40960 |
| Qwen3-0.6B | 28 | 16 | 8 | 128 | 40960 |

Source: `https://huggingface.co/Qwen/Qwen3-*/raw/main/config.json`

### MiniMax (Standard GQA + MTP)

| Model | Layers | Attn Heads | KV Heads | head_dim | rotary_dim | MTP Modules | Max Ctx |
|-------|--------|-----------|----------|----------|-----------|------------|---------|
| MiniMax M2 | 62 | 48 | 8 | 128 | 64 | 3 | 196608 |
| MiniMax M2.1 | 62 | 48 | 8 | 128 | 64 | 3 | 196608 |
| MiniMax M2.5 | 62 | 48 | 8 | 128 | 64 | 3 | 196608 |
| MiniMax M2.7 | 62 | 48 | 8 | 128 | 64 | 3 | 204800 |

Note: MiniMax models use Multi-Token Prediction (MTP) with 3 prediction modules, each with 1 transformer layer.

Source: `https://huggingface.co/MiniMaxAI/MiniMax-M2*/raw/main/config.json`

---

## Models Added in PR #7

### DeepSeek V3 / R1 (MLA)

| Model | Layers | kv_lora_rank | qk_rope_head_dim | qk_nope_head_dim | v_head_dim | KV Heads | Max Ctx |
|-------|--------|-------------|-------------------|------------------|-----------|----------|---------|
| DeepSeek V3 | 61 | 512 | 64 | 128 | 128 | 128 | 163840 |
| DeepSeek R1 | 61 | 512 | 64 | 128 | 128 | 128 | 163840 |

Source: `https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json`

### Llama 3.1 / 3.3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Llama 3.1 8B | 32 | 32 | 8 | 128 | 131072 |
| Llama 3.1 70B | 80 | 64 | 8 | 128 | 131072 |
| Llama 3.1 405B | 126 | 128 | 8 | 128 | 131072 |
| Llama 3.3 70B | 80 | 64 | 8 | 128 | 131072 |

Source: `https://huggingface.co/meta-llama/Llama-3.1-*-Instruct/raw/main/config.json`

### Qwen2.5 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Qwen2.5-72B | 80 | 64 | 8 | 128 | 131072 |
| Qwen2.5-32B | 64 | 40 | 8 | 128 | 131072 |
| Qwen2.5-14B | 48 | 40 | 8 | 128 | 131072 |
| Qwen2.5-7B | 28 | 28 | 4 | 128 | 131072 |

Source: `https://huggingface.co/Qwen/Qwen2.5-*-Instruct/raw/main/config.json`

### Mistral (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Mistral Large (2411) | 88 | 96 | 8 | 128 | 131072 |
| Mistral Small 24B | 40 | 32 | 8 | 128 | 32768 |

Source: `https://huggingface.co/mistralai/Mistral-Large-Instruct-2411/raw/main/config.json`

### Gemma 2 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Gemma 2 27B | 46 | 32 | 16 | 128 | 8192 |
| Gemma 2 9B | 42 | 16 | 8 | 256 | 8192 |

Source: `https://huggingface.co/google/gemma-2-*-it/raw/main/config.json`

### Gemma 3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Gemma 3 27B | 62 | 32 | 16 | 128 | 131072 |
| Gemma 3 12B | 48 | 16 | 8 | 256 | 131072 |

Note: Gemma 3 uses interleaved local/global attention (sliding_window=1024, global every ~4th layer). The `standard_gqa` formula gives a conservative (upper-bound) estimate.

Source: `https://huggingface.co/google/gemma-3-*-it/raw/main/config.json`

### Phi-4 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Phi-4 14B | 40 | 40 | 10 | 128 | 16384 |

Source: `https://huggingface.co/microsoft/phi-4/raw/main/config.json`

### Cohere Command R (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Command R+ | 64 | 96 | 8 | 128 | 131072 |
| Command R | 40 | 64 | 8 | 128 | 131072 |

Source: `https://huggingface.co/CohereForAI/c4ai-command-r-*/raw/main/config.json`

### Llama 4 (Standard GQA, iRoPE chunked attention)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx | Notes |
|-------|--------|-----------|----------|----------|---------|-------|
| Llama 4 Scout 17B-16E | 48 | 40 | 8 | 128 | 10485760 | Chunked attn (chunk=8192), 16 MoE experts |
| Llama 4 Maverick 17B-128E | 48 | 40 | 8 | 128 | 1048576 | Chunked attn (chunk=8192), 128 MoE experts |

Source: `https://huggingface.co/meta-llama/Llama-4-*/raw/main/config.json`

### Mixtral (Standard GQA, MoE)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx | Notes |
|-------|--------|-----------|----------|----------|---------|-------|
| Mixtral 8x7B | 32 | 32 | 8 | 128 | 32768 | 8 experts, top-2 |
| Mixtral 8x22B | 56 | 48 | 8 | 128 | 65536 | 8 experts, top-2 |

Source: `https://huggingface.co/mistralai/Mixtral-*/raw/main/config.json`

### Yi (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Yi-1.5-34B | 60 | 56 | 8 | 128 | 4096 |
| Yi-1.5-9B | 48 | 32 | 4 | 128 | 4096 |

Source: `https://huggingface.co/01-ai/Yi-1.5-*/raw/main/config.json`

### InternLM (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| InternLM2.5-20B | 48 | 48 | 8 | 128 | 32768 |
| InternLM3-8B | 32 | 32 | 8 | 128 | 32768 |

Source: `https://huggingface.co/internlm/internlm*/raw/main/config.json`

### DBRX (Standard GQA, MoE)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx | Notes |
|-------|--------|-----------|----------|----------|---------|-------|
| DBRX 132B | 40 | 48 | 8 | 128 | 32768 | 16 experts, top-4 |

Source: `https://huggingface.co/databricks/dbrx-instruct/raw/main/config.json`

### Phi-3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Phi-3-medium 14B | 40 | 40 | 10 | 128 | 131072 |
| Phi-3-small 7B | 32 | 32 | 8 | 128 | 131072 |
| Phi-3-mini 3.8B | 32 | 32 | 32 | 96 | 131072 |

Source: `https://huggingface.co/microsoft/Phi-3-*/raw/main/config.json`

### Qwen2 / Qwen2.5-Coder (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Qwen2-57B-A14B | 28 | 28 | 4 | 128 | 32768 |
| Qwen2.5-Coder-32B | 64 | 40 | 8 | 128 | 131072 |

Source: `https://huggingface.co/Qwen/Qwen2*/raw/main/config.json`

### Falcon (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Falcon-180B | 80 | 232 | 8 | 64 | 2048 |
| Falcon-40B | 60 | 128 | 8 | 64 | 2048 |

Source: `https://huggingface.co/tiiuae/falcon-*/raw/main/config.json`

### Nemotron (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Nemotron-4-340B | 96 | 96 | 8 | 192 | 4096 |

Source: `https://huggingface.co/nvidia/Nemotron-4-340B-Base/raw/main/config.json`

### Llama 2 (Standard MHA/GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Llama 2 7B | 32 | 32 | 32 | 128 | 4096 |
| Llama 2 13B | 40 | 40 | 40 | 128 | 4096 |
| Llama 2 70B | 80 | 64 | 8 | 128 | 4096 |

Note: Llama 2 7B/13B use Multi-Head Attention (MHA, kv_heads = attn_heads). Llama 2 70B uses Grouped Query Attention (GQA).

Source: `https://huggingface.co/meta-llama/Llama-2-*-chat-hf/raw/main/config.json`

### Llama 3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Llama 3 8B | 32 | 32 | 8 | 128 | 8192 |
| Llama 3 70B | 80 | 64 | 8 | 128 | 8192 |

Source: `https://huggingface.co/meta-llama/Meta-Llama-3-*-Instruct/raw/main/config.json`

### Llama 3.2 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Llama 3.2 1B | 16 | 32 | 8 | 64 | 131072 |
| Llama 3.2 3B | 28 | 24 | 8 | 128 | 131072 |

Source: `https://huggingface.co/meta-llama/Llama-3.2-*-Instruct/raw/main/config.json`

### CodeLlama (Standard MHA/GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| CodeLlama 7B | 32 | 32 | 32 | 128 | 16384 |
| CodeLlama 13B | 40 | 40 | 40 | 128 | 16384 |
| CodeLlama 34B | 48 | 64 | 8 | 128 | 16384 |

Note: CodeLlama 7B/13B use MHA. CodeLlama 34B uses GQA.

Source: `https://huggingface.co/codellama/CodeLlama-*-Instruct-hf/raw/main/config.json`

### Mistral 7B / Nemo (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Mistral 7B v0.3 | 32 | 32 | 8 | 128 | 32768 |
| Mistral Nemo 12B | 40 | 32 | 8 | 128 | 131072 |

Source: `https://huggingface.co/mistralai/Mistral-*/raw/main/config.json`

### DeepSeek V2 / V2.5 / V2-Lite (MLA)

| Model | Layers | kv_lora_rank | qk_rope_head_dim | qk_nope_head_dim | v_head_dim | KV Heads | Max Ctx |
|-------|--------|-------------|------------------|------------------|-----------|----------|---------|
| DeepSeek V2 | 60 | 512 | 64 | 128 | 128 | 128 | 163840 |
| DeepSeek V2.5 | 60 | 512 | 64 | 128 | 128 | 128 | 163840 |
| DeepSeek V2-Lite | 27 | 512 | 64 | 128 | 128 | 16 | 163840 |

Source: `https://huggingface.co/deepseek-ai/DeepSeek-V2*/raw/main/config.json`

### StarCoder2 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| StarCoder2-3B | 30 | 24 | 2 | 128 | 16384 |
| StarCoder2-7B | 32 | 36 | 4 | 128 | 16384 |
| StarCoder2-15B | 40 | 48 | 4 | 128 | 16384 |

Source: `https://huggingface.co/bigcode/starcoder2-*/raw/main/config.json`

### Qwen2.5 Small Models (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Qwen2.5-3B | 36 | 16 | 2 | 128 | 32768 |
| Qwen2.5-1.5B | 28 | 12 | 2 | 128 | 32768 |
| Qwen2.5-0.5B | 24 | 14 | 2 | 64 | 32768 |

Source: `https://huggingface.co/Qwen/Qwen2.5-*-Instruct/raw/main/config.json`

### Gemma 2 2B (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Gemma 2 2B | 26 | 8 | 4 | 256 | 8192 |

Source: `https://huggingface.co/google/gemma-2-2b-it/raw/main/config.json`

### Baichuan2 (Standard MHA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Baichuan2-7B | 32 | 32 | 32 | 128 | 4096 |
| Baichuan2-13B | 40 | 40 | 40 | 128 | 4096 |

Note: Baichuan2 uses Multi-Head Attention (MHA, kv_heads = attn_heads).

Source: `https://huggingface.co/baichuan-inc/Baichuan2-*-Chat/raw/main/config.json`

### Jamba (Hybrid SSM+Attention)

| Model | Attn Layers | Attn Heads | KV Heads | head_dim | Max Ctx | Notes |
|-------|-------------|-----------|----------|----------|---------|-------|
| Jamba v0.1 (52B MoE) | 4 | 32 | 8 | 128 | 262144 | 32 total layers, attn_layer_period=8, 16 MoE experts |

Note: Jamba is a hybrid SSM+Attention model. Only attention layers produce KV cache. The `num_hidden_layers` field is set to the attention layer count (4 out of 32 total) for correct KV cache calculation.

Source: `https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini/raw/main/config.json`

### Phi-4-mini (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Phi-4-mini 3.8B | 32 | 24 | 8 | 128 | 131072 |

Source: `https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/config.json`

---

## Notes

- **MoE models**: KV cache is independent of MoE — only attention parameters matter.
- **Jamba (hybrid SSM+Attention)**: Only attention layers produce KV cache. Jamba v0.1 has 4 attention layers out of 32 total (attn_layer_period=8). The `num_hidden_layers` field is set to the attention layer count for correct KV cache calculation using `standard_gqa`.
- **Llama 2 7B/13B, CodeLlama 7B/13B, Baichuan2**: Use Multi-Head Attention (MHA) where num_key_value_heads = num_attention_heads. The `standard_gqa` formula handles this correctly (GQA is a generalization of MHA).
- **Llama 4 (iRoPE)**: Uses chunked attention (chunk_size=8192). Local-attention layers only cache chunk_size tokens. `standard_gqa` gives upper-bound estimate.
- **Gemma 3**: Uses interleaved sliding window (1024) + global attention. `standard_gqa` gives upper-bound estimate.
