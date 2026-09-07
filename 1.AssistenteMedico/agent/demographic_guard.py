"""Checagem de coerencia etaria da resposta do assistente.

Rede de seguranca sobre um erro especifico e de alto impacto clinico: a resposta
descreve conduta correta para o DIAGNOSTICO e errada para o PACIENTE, por supor
uma faixa etaria que nao e a dele. O caso observado foi uma pergunta sobre o
procedimento indicado para um paciente de 66 anos com hernia inguinal, respondida
com a conduta da hernia inguinal pediatrica.

## Por que esse erro acontece

A causa primaria e a ausencia do dado no prompt: idade e sexo nao constavam do
contexto clinico montado por agent/tools.py::montar_contexto_clinico, entao o
modelo preenchia a lacuna com a faixa etaria mais frequente na literatura do
diagnostico. Isso foi corrigido na origem — o contexto agora abre com idade e
sexo. Este modulo e a segunda camada: com o dado presente no prompt, um modelo
pequeno ainda pode ignora-lo, e a consequencia de nao perceber e alta demais para
depender de uma unica defesa.

## O que este modulo faz e o que nao faz

Ele NAO bloqueia a resposta nem tenta reescreve-la — mesma politica do guardrail
de prescricao (agent/guardrails.py): sinaliza para validacao humana e anexa a
ressalva, deixando a decisao com o medico. Uma resposta pode legitimamente
mencionar outra faixa etaria para contrastar ("diferente do que ocorre em
criancas, no adulto..."), e um filtro que descartasse esses casos removeria
informacao clinica correta. Sinalizar cobre o erro sem esse custo.

A deteccao e lexica e deliberadamente simples: um conjunto pequeno de termos de
faixa etaria, cada um com o intervalo de idade que ele implica. Nao ha inferencia
semantica nem modelo envolvido — o resultado e explicavel a partir do termo exato
que disparou, que vai para o log de auditoria.

A regra nao e "mencionou uma faixa que nao e a do paciente". E "mencionou faixas
etarias e NENHUMA delas contem a idade do paciente". A diferenca importa: uma
resposta que diz "diferente do que ocorre em criancas, no idoso a conduta e..."
cita uma faixa incompativel, mas demonstra estar orientada pela idade certa ao
citar tambem a compativel. O erro que se quer pegar e a resposta inteiramente
ancorada na faixa errada, e essa nao tem nenhum termo compativel. Sinalizar a
primeira junto com a segunda produziria alarme em respostas corretas — e um
guardrail que dispara em resposta boa deixa de ser lido.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Termos de faixa etaria e o intervalo (min, max) de anos que cada um implica.
# Intervalos deliberadamente largos nas bordas para nao sinalizar por diferenca
# de um ano em definicoes que variam entre fontes.
_FAIXAS_ETARIAS: list[tuple[str, int, int]] = [
    (r"rec[eé]m[- ]nascid[oa]s?", 0, 0),
    (r"neonat[oa]l?s?", 0, 0),
    (r"lactentes?", 0, 2),
    (r"crian[cç]as?", 0, 12),
    (r"infantis?", 0, 12),
    (r"infantil", 0, 12),
    (r"pedi[aá]tric[oa]s?", 0, 17),
    (r"pediatria", 0, 17),
    (r"adolescentes?", 10, 19),
    (r"menor(?:es)? de idade", 0, 17),
    (r"idos[oa]s?", 60, 130),
    (r"geri[aá]tric[oa]s?", 60, 130),
]

_DISCLAIMER = (
    "\n\n**Atencao:** esta resposta menciona uma faixa etaria que nao corresponde "
    "a idade registrada para este paciente. Confira se a conduta descrita se aplica "
    "ao caso antes de qualquer decisao."
)


@dataclass
class CoerenciaEtariaResult:
    is_flagged: bool
    idade_paciente: int | None = None
    termos_incoerentes: list[str] = field(default_factory=list)
    safe_response: str = ""


def _normalizar(texto: str) -> str:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", texto or "") if not unicodedata.combining(c)
    )
    return sem_acento.lower()


def check_age_coherence(resposta: str, idade: int | None) -> CoerenciaEtariaResult:
    """Procura na resposta termos de faixa etaria incompativeis com `idade`.

    idade None (paciente sem data de nascimento registrada) desativa a checagem:
    sem a idade real nao ha com o que comparar, e sinalizar seria arbitrario.
    """
    if idade is None or not resposta:
        return CoerenciaEtariaResult(
            is_flagged=False, idade_paciente=idade, safe_response=resposta or ""
        )

    # A busca roda sobre o texto sem acento para que um unico padrao cubra as
    # duas grafias, mas o termo reportado no log e o que aparece no texto
    # original — e ele que o revisor vai procurar na resposta.
    normalizado = _normalizar(resposta)
    padrao_sem_acento = {
        padrao: _normalizar(padrao).replace("\\", "") for padrao, _, _ in _FAIXAS_ETARIAS
    }

    incoerentes: list[str] = []
    tem_faixa_compativel = False
    for padrao, minimo, maximo in _FAIXAS_ETARIAS:
        match = re.search(padrao_sem_acento[padrao], normalizado)
        if not match:
            continue
        if minimo <= idade <= maximo:
            tem_faixa_compativel = True
        else:
            incoerentes.append(match.group(0))

    # Ver "A regra nao e..." no docstring do modulo: a presenca de ao menos uma
    # faixa compativel indica que a resposta esta orientada pela idade correta e
    # apenas contrasta com outra faixa.
    if not incoerentes or tem_faixa_compativel:
        return CoerenciaEtariaResult(
            is_flagged=False, idade_paciente=idade, safe_response=resposta
        )

    # dict.fromkeys preserva a ordem de aparicao e remove duplicatas — varios
    # padroes podem casar com o mesmo trecho (ex. "criancas" e "infantil").
    return CoerenciaEtariaResult(
        is_flagged=True,
        idade_paciente=idade,
        termos_incoerentes=list(dict.fromkeys(incoerentes)),
        safe_response=resposta + _DISCLAIMER,
    )
