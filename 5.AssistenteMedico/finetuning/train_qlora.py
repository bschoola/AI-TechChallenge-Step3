"""Fine-tuning do modelo base com o dataset interno sintetico — QLoRA em GPU, ou
LoRA "cru" (sem quantizacao) como fallback de CPU.

RECOMENDADO: rodar no Google Colab (GPU T4 gratuita) — ver
finetuning/train_qlora_colab.ipynb. E MUITO mais rapido.

FALLBACK DE CPU: bitsandbytes 4-bit (QLoRA) exige CUDA, entao sem GPU este script
carrega o modelo sem quantizacao (fp32/bf16, ver CPU_DTYPE em config.py) e aplica
LoRA por cima — tecnicamente funciona com RAM suficiente (Qwen2.5-3B em bf16 usa
~6GB so para os pesos), mas e ordens de magnitude mais lento que GPU: um treino que
leva minutos numa T4 pode levar horas numa CPU, mesmo com o dataset sintetico atual
sendo pequeno. Se quiser algo mais rapido no seu proprio hardware, troque
BASE_MODEL_NAME em config.py para um modelo menor (Qwen2.5-1.5B-Instruct ou
Qwen2.5-0.5B-Instruct) antes de treinar — ver PLANO_Fase3.md secao 2.1.

Uso:
    python finetuning/train_qlora.py

Alternativa mais rapida para Colab free: substituir os imports de transformers/peft
por `unsloth.FastLanguageModel`, que reduz uso de VRAM e acelera o treino em ~2x
mantendo a mesma logica de dataset e hiperparametros deste arquivo.
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import (
    ADAPTER_OUTPUT_DIR,
    BASE_MODEL_NAME,
    BNB_COMPUTE_DTYPE,
    CPU_DTYPE,
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
    SYSTEM_PROMPT,
)


def _resolve_compute_dtype() -> torch.dtype:
    """Resolve o dtype de computo do bitsandbytes (so usado com GPU). 'auto' escolhe
    bfloat16 em GPUs que suportam nativamente (Ampere+, ex. A100) ou float16 caso
    contrario (Turing, ex. a propria T4 do Colab free, que NAO suporta bf16
    nativamente) — usar bf16 numa T4 funciona, mas sem o ganho de performance.
    """
    if BNB_COMPUTE_DTYPE == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, BNB_COMPUTE_DTYPE)


def load_base_model_and_tokenizer():
    """Carrega o modelo base + tokenizer.

    Com GPU CUDA disponivel: QLoRA de verdade, modelo quantizado em 4-bit
    (bitsandbytes). Sem GPU: fallback de CPU, modelo carregado sem quantizacao no
    dtype definido em config.CPU_DTYPE (mais RAM, muito mais lento, mas funciona).
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        compute_dtype = _resolve_compute_dtype()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print(
            "Aviso: nenhuma GPU CUDA disponivel. Carregando o modelo SEM "
            f"quantizacao (dtype={CPU_DTYPE}) para rodar em CPU — isso vai ser "
            "bem mais lento que no Colab (ver finetuning/train_qlora_colab.ipynb). "
            "Considere um modelo menor em config.py::BASE_MODEL_NAME se a "
            "velocidade for um problema."
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=getattr(torch, CPU_DTYPE),
            device_map="cpu",
        )

    return model, tokenizer


def apply_lora(model):
    """Aplica LoRA nas camadas de atencao (ver LORA_TARGET_MODULES em config.py).

    Em modelo quantizado (GPU/4-bit), passa por prepare_model_for_kbit_training
    primeiro (padrao QLoRA). Em modelo "cru" (fallback de CPU), habilita gradient
    checkpointing na mao para economizar memoria, sem a preparacao especifica de
    k-bit (que nao se aplica quando nao ha quantizacao).
    """
    is_quantized = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

    if is_quantized:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_dataset_splits():
    """Carrega DATASET_TRAIN_PATH / DATASET_VAL_PATH gerados por prepare_dataset.py."""
    if not DATASET_TRAIN_PATH.exists() or not DATASET_VAL_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em {DATASET_TRAIN_PATH} ou {DATASET_VAL_PATH}. "
            "Rode 'python finetuning/prepare_dataset.py' primeiro."
        )
    dataset = load_dataset(
        "json",
        data_files={"train": str(DATASET_TRAIN_PATH), "validation": str(DATASET_VAL_PATH)},
    )
    return dataset["train"], dataset["validation"]


def build_prompt(example: dict, tokenizer) -> str:
    """Formata um exemplo {'instruction', 'input', 'output'} no template de chat do
    modelo, usando o MESMO system prompt aplicado em runtime (config.SYSTEM_PROMPT,
    reaproveitado por rag/chain.py) — o fine-tuning ensina o modelo a responder
    dentro dos mesmos limites que o guardrail de runtime depois verifica
    (agent/guardrails.py), nao um comportamento diferente do que sera validado.
    """
    user_content = example["instruction"]
    if example.get("input"):
        user_content = f"{user_content}\n\n{example['input']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def train() -> None:
    """Orquestra o treino com trl.SFTTrainer e salva o adapter em ADAPTER_OUTPUT_DIR."""
    model, tokenizer = load_base_model_and_tokenizer()
    model = apply_lora(model)
    train_ds, val_ds = load_dataset_splits()

    train_ds = train_ds.map(lambda ex: {"text": build_prompt(ex, tokenizer)})
    val_ds = val_ds.map(lambda ex: {"text": build_prompt(ex, tokenizer)})

    # Os flags bf16/fp16 do Trainer ligam autocast de precisao mista para GPU — em
    # CPU o modelo ja foi carregado direto no dtype de config.CPU_DTYPE (sem
    # quantizacao), entao deixamos ambos False e o treino roda nesse dtype nativo.
    if torch.cuda.is_available():
        compute_dtype = _resolve_compute_dtype()
        use_bf16 = compute_dtype == torch.bfloat16
        use_fp16 = compute_dtype == torch.float16
    else:
        use_bf16 = use_fp16 = False

    training_args = SFTConfig(
        output_dir=str(ADAPTER_OUTPUT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    print(f"Adapter LoRA salvo em {ADAPTER_OUTPUT_DIR}")


if __name__ == "__main__":
    train()
