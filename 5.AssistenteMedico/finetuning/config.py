"""Configurações centrais do fine-tuning — equivalente ao config.py de 1.GenAIPrediction."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Modelo base ---
# Recomendado: sem gate de licença, roda em QLoRA 4-bit no Colab free (T4, ~15GB VRAM).
# Ver PLANO_Fase3.md secao 2.1 para alternativas (Qwen2.5-1.5B se a sessao do Colab cair).
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# --- LoRA / QLoRA ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LOAD_IN_4BIT = True
BNB_COMPUTE_DTYPE = "bfloat16"

# --- Treino ---
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
RANDOM_STATE = 42

# --- Paths ---
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATASET_TRAIN_PATH = BASE_DIR / "data" / "processed" / "dataset_train.jsonl"
DATASET_VAL_PATH = BASE_DIR / "data" / "processed" / "dataset_val.jsonl"
ADAPTER_OUTPUT_DIR = BASE_DIR / "finetuning" / "adapters" / "cancer-assistant-lora"

VAL_SPLIT_RATIO = 0.15
