"""Fine-tuning QLoRA do modelo base com o dataset interno sintetico.

Pensado para rodar em Colab (GPU T4 gratuita). Localmente sem GPU, reduza
NUM_TRAIN_EPOCHS e o tamanho do dataset apenas para validar que o pipeline roda
(nao espere qualidade real sem GPU).

Fluxo (ver PLANO_Fase3.md secao 2.1 e 4):
    1. Carregar modelo base em 4-bit (bitsandbytes)
    2. Aplicar LoRA (peft) nas camadas de atencao
    3. Treinar com trl.SFTTrainer sobre dataset_train.jsonl
    4. Salvar apenas o adapter LoRA em ADAPTER_OUTPUT_DIR

Uso:
    python finetuning/train_qlora.py

Alternativa mais rapida para Colab free: substituir os imports de transformers/peft
por `unsloth.FastLanguageModel`, que reduz uso de VRAM e acelera o treino em ~2x
mantendo a mesma logica de dataset e hiperparametros deste arquivo.
"""

from config import (
    ADAPTER_OUTPUT_DIR,
    BASE_MODEL_NAME,
    DATASET_TRAIN_PATH,
    DATASET_VAL_PATH,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
)


def load_base_model_and_tokenizer():
    """Carrega o modelo base em 4-bit + tokenizer.

    TODO: usar BitsAndBytesConfig(load_in_4bit=LOAD_IN_4BIT, ...) e
    AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=...).
    """
    raise NotImplementedError


def apply_lora(model):
    """Aplica configuracao LoRA (peft.LoraConfig + get_peft_model)."""
    raise NotImplementedError


def load_dataset_splits():
    """Carrega DATASET_TRAIN_PATH / DATASET_VAL_PATH via datasets.load_dataset('json', ...)."""
    raise NotImplementedError


def build_prompt(example: dict) -> str:
    """Formata um exemplo {'instruction', 'input', 'output'} no template de chat do modelo.

    TODO: usar tokenizer.apply_chat_template com o system prompt do assistente
    (mesmo tom/limites definidos em agent/guardrails.py) + instruction/input do exemplo,
    e output como resposta esperada.
    """
    raise NotImplementedError


def train() -> None:
    """Orquestra o treino com trl.SFTTrainer e salva o adapter em ADAPTER_OUTPUT_DIR."""
    model, tokenizer = load_base_model_and_tokenizer()
    model = apply_lora(model)
    train_ds, val_ds = load_dataset_splits()

    # TODO: instanciar trl.SFTTrainer(model=model, tokenizer=tokenizer,
    #   train_dataset=train_ds, eval_dataset=val_ds,
    #   args=TrainingArguments(num_train_epochs=NUM_TRAIN_EPOCHS, ...))
    # e chamar trainer.train() + trainer.save_model(str(ADAPTER_OUTPUT_DIR))
    raise NotImplementedError


if __name__ == "__main__":
    train()
