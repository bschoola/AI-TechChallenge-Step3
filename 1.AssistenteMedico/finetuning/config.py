"""Configurações centrais do fine-tuning (modelo base, hiperparâmetros LoRA, paths)."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Modelo base ---
# Recomendado: sem gate de licença, roda em QLoRA 4-bit no Colab free (T4, ~15GB VRAM).
# Ver PLANO_Fase3.md secao 2.1 para alternativas (Qwen2.5-1.5B se a sessao do Colab cair).
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# --- LoRA / QLoRA ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LOAD_IN_4BIT = True
# "auto" deixa o script escolher bfloat16 (GPUs Ampere+, ex. A100) ou float16
# (GPUs Turing, ex. a T4 do Colab free — que NAO suporta bf16 nativamente).
# So force um valor fixo aqui ("bfloat16"/"float16") se souber exatamente em que
# GPU vai rodar. So se aplica quando ha GPU disponivel (ver CPU_DTYPE abaixo).
BNB_COMPUTE_DTYPE = "auto"

# Dtype usado quando NAO ha GPU disponivel (fallback de CPU — ver train_qlora.py).
# bitsandbytes 4-bit exige CUDA, entao sem GPU o script carrega o modelo "cru"
# (sem quantizacao) neste dtype. bfloat16 usa metade da RAM de float32 (~6GB vs
# ~12GB para o Qwen2.5-3B) e funciona em CPUs modernas; troque para "float32" se
# sua CPU/PyTorch reclamar de operacao nao suportada em bf16.
CPU_DTYPE = "bfloat16"

# --- Treino ---
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024  # usado como SFTConfig.max_length em train_qlora.py
RANDOM_STATE = 42

# System prompt usado tanto no fine-tuning (finetuning/train_qlora.py::build_prompt)
# quanto no runtime (rag/chain.py) — o modelo deve aprender a responder dentro dos
# MESMOS limites que o guardrail de runtime reforca (agent/guardrails.py), nao um
# comportamento diferente do que e validado depois.
SYSTEM_PROMPT = (
    "Voce e um assistente medico de apoio a decisao clinica do Hospital vEinstein. "
    "Use as informacoes de contexto fornecidas para responder. Se o contexto nao for "
    "suficiente, diga isso explicitamente. NUNCA prescreva um tratamento diretamente: "
    "sempre enquadre sugestoes como recomendacao a ser validada por um medico "
    "responsavel. Quando citar um protocolo, indique o nome dele na resposta."
)

# --- Paths ---
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATASET_TRAIN_PATH = BASE_DIR / "data" / "processed" / "dataset_train.jsonl"
DATASET_VAL_PATH = BASE_DIR / "data" / "processed" / "dataset_val.jsonl"
ADAPTER_OUTPUT_DIR = BASE_DIR / "finetuning" / "adapters" / "cancer-assistant-lora"

VAL_SPLIT_RATIO = 0.15
