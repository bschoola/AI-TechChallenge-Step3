"""Guardrail de escopo do chat: decide se a pergunta livre do medico e sobre o
dominio clinico ANTES de montar contexto do paciente e chamar o LLM.

Motivacao (ver discussao com o usuario): perguntas fora do dominio medico (ex.:
"onde ir na praia perto de Sao Paulo?") faziam o modelo tentar recusar e, no
mesmo texto, puxar pedacos do contexto do paciente que estava no prompt -- um
modelo pequeno fine-tuned em QA factual tende a "completar" em vez de recusar
de forma seca. A causa raiz e estrutural: o pipeline sempre injeta o contexto
clinico completo (agent/tools.py::montar_contexto_clinico) e deixa a propria
LLM decidir, em texto livre, se responde ou nao.

## Historico: por que similaridade absoluta nao funcionou

A primeira versao deste modulo comparava a pergunta so contra ancoras "dentro
do escopo" com um limiar absoluto (0.75). Em teste real (ver
logs/audit.jsonl), perguntas obviamente fora de escopo ("Devo trocar a vela de
ignicao do meu carro a cada 12 meses?", "Qual melhor bola de futebol para
comprar?", "E um bom momento para viajar para a Jamaica?") tiveram similaridade
0.82-0.85 contra as ancoras clinicas -- ACIMA do limiar de 0.75, ou seja,
classificadas (erradamente) como dentro do escopo.

Causa: o multilingual-e5-small tem um "piso" de similaridade alto (~0.8) entre
quaisquer duas frases curtas em portugues, so pela forma de pergunta/registro
formal, independente do assunto. Um limiar absoluto contra um unico conjunto de
ancoras nao separa esse piso do sinal real de topico.

Correcao: comparacao RELATIVA (abordagem tipo "semantic router"). Alem das
ancoras dentro do escopo, ha um conjunto de ancoras FORA do escopo (viagem,
carro, esporte, clima etc. -- incluindo os proprios casos que falharam acima).
A pergunta e classificada como dentro do escopo so quando a maior similaridade
contra as ancoras clinicas supera a maior similaridade contra as ancoras fora
de escopo (por uma margem minima, SCOPE_MARGIN) -- isso cancela o piso comum
aos dois lados e usa so o sinal de topico.

Vantagens desta abordagem sobre deixar o proprio LLM decidir:
- Deterministico e barato (sem chamada extra ao modelo fine-tuned/pipeline de
  geracao, que e o componente mais lento do grafo).
- Reaproveita infraestrutura que ja existe (rag/embeddings.py), sem dependencia
  nova.
- As similaridades ficam registradas no log de auditoria
  (agent/nodes.py::log_auditoria), o que ajuda a recalibrar SCOPE_MARGIN com
  casos reais e e evidencia de explainability para o relatorio tecnico.

Nao substitui os guardrails existentes (agent/guardrails.py) -- e uma camada
anterior, que evita que a LLM sequer tente responder o que nao deveria.
"""

import math
from dataclasses import dataclass

# Perguntas-ancora do que E escopo do assistente: apoio a decisao clinica sobre
# o paciente (historico, anamnese, exames, protocolos, conduta). Nao precisa
# ser exaustivo -- a similaridade de embeddings generaliza para variacoes,
# sinonimos e perguntas informais dentro do mesmo dominio semantico.
ANCORAS_DENTRO_DO_ESCOPO = [
    "Qual o historico medico deste paciente?",
    "Quais exames o paciente ja realizou e quais estao pendentes?",
    "Qual o protocolo de tratamento indicado para este caso?",
    "Quais sintomas foram relatados na anamnese?",
    "Existe alguma interacao medicamentosa a considerar?",
    "Qual a hipotese diagnostica mais provavel para este quadro?",
    "Quais cuidados devo ter no acompanhamento deste paciente?",
    "Pode explicar o resultado deste exame?",
    "Qual a conduta recomendada para esse quadro clinico?",
    "Esse paciente tem alguma alergia ou comorbidade registrada?",
    "Quais sao os efeitos colaterais esperados desse tratamento?",
    "O que os sinais vitais do paciente indicam?",
]

# Perguntas-ancora do que E FORA do escopo -- cobrindo as categorias que ja
# provaram furar um limiar absoluto (ver historico acima): viagem/turismo,
# esporte/compras, veiculos, clima, entretenimento, conhecimento geral,
# programacao, culinaria, financas pessoais.
ANCORAS_FORA_DO_ESCOPO = [
    "Qual o melhor lugar para viajar de ferias?",
    "Onde fica a praia mais bonita perto de Sao Paulo?",
    "Qual time vai ganhar o campeonato de futebol?",
    "Qual a melhor bola de futebol para comprar?",
    "Devo trocar a vela de ignicao do carro a cada quanto tempo?",
    "Como faco a manutencao do meu carro?",
    "Qual vai ser a previsao do tempo para amanha?",
    "Qual a capital da Franca?",
    "Como programar em Python?",
    "Qual o melhor filme para assistir hoje?",
    "Como faco para investir na bolsa de valores?",
    "Qual a melhor receita de bolo de chocolate?",
]

# Margem minima (similaridade_dentro - similaridade_fora) para considerar a
# pergunta dentro do escopo. 0.0 = decisao puramente relativa (a pergunta
# precisa estar so um pouco mais perto do lado clinico). Vale recalibrar
# olhando `similaridade_escopo`/`similaridade_escopo_dentro`/
# `similaridade_escopo_fora` no logs/audit.jsonl depois de uso real (ver
# agent/nodes.py::log_auditoria) -- especialmente testando perguntas clinicas
# genuinas, que ainda nao temos dado real de producao (so temos dado real do
# lado "fora de escopo" ate agora).
SCOPE_MARGIN = 0.0

FORA_DE_ESCOPO_RESPOSTA = (
    "Nao tenho como ajudar com isso. Meu papel aqui e apoiar decisoes clinicas "
    "sobre o paciente -- historico, anamnese, exames, protocolos e conduta. "
    "Para assuntos fora da area medica, procure outra fonte."
)

_embeddings = None
_ancoras_dentro_vecs: list[list[float]] | None = None
_ancoras_fora_vecs: list[list[float]] | None = None


def _get_embeddings():
    # Import tardio (mesmo padrao de rag/chain.py::load_llm_and_tokenizer): evita
    # puxar langchain_huggingface/sentence-transformers so por importar este
    # modulo -- os testes de logica pura (cosine_similarity, classify_scope com
    # embeddings falsos) nao precisam da stack pesada instalada.
    global _embeddings
    if _embeddings is None:
        from rag.embeddings import load_embeddings
        _embeddings = load_embeddings()
    return _embeddings


def _get_ancoras_dentro_vecs() -> list[list[float]]:
    global _ancoras_dentro_vecs
    if _ancoras_dentro_vecs is None:
        _ancoras_dentro_vecs = _get_embeddings().embed_documents(ANCORAS_DENTRO_DO_ESCOPO)
    return _ancoras_dentro_vecs


def _get_ancoras_fora_vecs() -> list[list[float]]:
    global _ancoras_fora_vecs
    if _ancoras_fora_vecs is None:
        _ancoras_fora_vecs = _get_embeddings().embed_documents(ANCORAS_FORA_DO_ESCOPO)
    return _ancoras_fora_vecs


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similaridade de cosseno pura em Python (sem numpy) -- os vetores de
    embedding aqui sao curtos (384 dims no multilingual-e5-small) e isso roda
    uma vez por pergunta, entao nao vale adicionar uma dependencia so por
    causa disso.
    """
    denom = _norm(a) * _norm(b)
    if denom == 0:
        return 0.0
    return _dot(a, b) / denom


def _max_similarity(query_vec: list[float], ancoras: list[list[float]]) -> float:
    return max(cosine_similarity(query_vec, ancora) for ancora in ancoras)


@dataclass
class ScopeResult:
    dentro_do_escopo: bool
    similaridade_dentro: float  # maior similaridade contra ancoras clinicas
    similaridade_fora: float  # maior similaridade contra ancoras fora de escopo
    margem: float  # similaridade_dentro - similaridade_fora


def classify_scope(question: str, margin: float = SCOPE_MARGIN) -> ScopeResult:
    """Classifica a pergunta do chat por similaridade RELATIVA de embeddings
    (ver "Historico" no docstring do modulo para o porque de nao usar limiar
    absoluto). Usado por agent/nodes.py::classificar_escopo.
    """
    query_vec = _get_embeddings().embed_query(question)
    sim_dentro = _max_similarity(query_vec, _get_ancoras_dentro_vecs())
    sim_fora = _max_similarity(query_vec, _get_ancoras_fora_vecs())
    margem = sim_dentro - sim_fora
    return ScopeResult(
        dentro_do_escopo=margem >= margin,
        similaridade_dentro=sim_dentro,
        similaridade_fora=sim_fora,
        margem=margem,
    )
