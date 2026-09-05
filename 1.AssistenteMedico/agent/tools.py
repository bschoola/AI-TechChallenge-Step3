"""Tool de consulta ao 'prontuario' — banco estruturado simulado em SQLite.

Contem tambem o script de inicializacao do banco com dados sinteticos, para que o
projeto rode fim-a-fim sem depender de um sistema hospitalar real.
"""

import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "prontuarios_mock.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY,
    nome_ficticio TEXT NOT NULL,
    historico TEXT,
    sexo TEXT,
    data_nascimento TEXT
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL,
    tipo TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pendente', 'concluido')),
    data_solicitacao TEXT,
    data_realizacao TEXT,
    resultado TEXT,
    medico_solicitante TEXT,
    observacoes TEXT,
    FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);
"""

# Colunas adicionadas depois da primeira versao do schema (data_solicitacao,
# data_realizacao, resultado, medico_solicitante, observacoes). Quem ja tinha um
# prontuarios_mock.db gerado antes dessa mudanca so tem (id, paciente_id, tipo,
# status) na tabela — "CREATE TABLE IF NOT EXISTS" nao adiciona coluna em tabela
# existente, entao migramos na mao via ALTER TABLE (idempotente: so adiciona o que
# ainda nao existe).
_EXAME_COLUNAS_NOVAS = {
    "data_solicitacao": "TEXT",
    "data_realizacao": "TEXT",
    "resultado": "TEXT",
    "medico_solicitante": "TEXT",
    "observacoes": "TEXT",
}

# Mesma logica de migracao para as colunas novas de `pacientes` (sexo,
# data_nascimento). Note que "idade" NAO e uma coluna: fica desatualizada com o
# tempo se for armazenada (paciente faz aniversario, dado nao muda sozinho) — em
# vez disso e sempre calculada na hora a partir de data_nascimento (ver
# _calcular_idade / list_pacientes), garantindo que nunca fica errada.
_PACIENTE_COLUNAS_NOVAS = {
    "sexo": "TEXT",
    "data_nascimento": "TEXT",
}

# Dados de exemplo — todos ficticios, para permitir demonstrar o fluxo no video.
_SEED_PACIENTES = [
    (1, "Paciente Ficticio A", "Historico de hipertensao controlada."),
    (2, "Paciente Ficticio B", "Sem comorbidades registradas."),
]

# Sexo/data de nascimento dos pacientes semente, por id. Preenchido so quando a
# coluna ainda estiver vazia (mesmo padrao de _seed_exame_detalhes) — funciona
# tanto num banco novo quanto num banco migrado de uma versao anterior do schema.
_SEED_PACIENTE_DETALHES = {
    1: {  # Paciente Ficticio A — perfil consistente com rastreio de rotina (mamografia)
        "sexo": "Feminino",
        "data_nascimento": "1972-03-14",
    },
    2: {  # Paciente Ficticio B — perfil consistente com investigacao de nodulo (ultrassom)
        "sexo": "Feminino",
        "data_nascimento": "1989-11-02",
    },
}

_SEED_EXAMES = [
    (1, "Mamografia", "pendente"),
    (1, "Hemograma completo", "concluido"),
    (2, "Ultrassom", "pendente"),
]

# Detalhe dos exames semente, por id (a ordem de insercao acima determina o id
# autoincrement: Mamografia=1, Hemograma completo=2, Ultrassom=3, num banco novo).
# Preenchido so quando a coluna ainda esta vazia (ver _seed_exame_detalhes) — nao
# sobrescreve nada que ja tenha sido preenchido de outra forma.
_SEED_EXAME_DETALHES = {
    1: {  # Mamografia — paciente 1, pendente
        "data_solicitacao": "2026-08-20",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Camila Nogueira (Ginecologia)",
        "observacoes": (
            "Rastreamento de rotina anual. Paciente relata historico familiar de "
            "cancer de mama a confirmar na consulta de retorno."
        ),
    },
    2: {  # Hemograma completo — paciente 1, concluido
        "data_solicitacao": "2026-08-10",
        "data_realizacao": "2026-08-12",
        "resultado": (
            "Hemoglobina 13.2 g/dL, leucocitos 6.800/mm3, plaquetas 245.000/mm3 — "
            "todos os valores dentro da faixa de referencia."
        ),
        "medico_solicitante": "Dr. Rafael Torres (Clinica Geral)",
        "observacoes": "Exame de rotina solicitado antes da consulta de retorno; sem alteracoes relevantes.",
    },
    3: {  # Ultrassom — paciente 2, pendente
        "data_solicitacao": "2026-08-22",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Paulo Mendes (Radiologia)",
        "observacoes": "Investigacao de nodulo palpavel identificado em exame clinico de rotina.",
    },
}


def _calcular_idade(data_nascimento: str | None, hoje: date | None = None) -> int | None:
    """Calcula a idade em anos completos a partir de data_nascimento (formato
    YYYY-MM-DD). Retorna None se data_nascimento nao estiver preenchida.
    """
    if not data_nascimento:
        return None
    hoje = hoje or date.today()
    ano, mes, dia = (int(parte) for parte in data_nascimento.split("-"))
    idade = hoje.year - ano
    if (hoje.month, hoje.day) < (mes, dia):
        idade -= 1
    return idade


def _migrar_schema_pacientes(conn: sqlite3.Connection) -> None:
    """Mesma ideia de _migrar_schema_exames, para as colunas novas de `pacientes`
    (sexo, data_nascimento) num banco criado por uma versao anterior do schema.
    """
    colunas_existentes = {row[1] for row in conn.execute("PRAGMA table_info(pacientes)")}
    for coluna, tipo_sql in _PACIENTE_COLUNAS_NOVAS.items():
        if coluna not in colunas_existentes:
            conn.execute(f"ALTER TABLE pacientes ADD COLUMN {coluna} {tipo_sql}")


def _seed_paciente_detalhes(conn: sqlite3.Connection) -> None:
    """Preenche sexo/data_nascimento dos pacientes semente quando ainda estiverem
    vazios (COALESCE mantem o valor existente) — mesmo padrao de
    _seed_exame_detalhes.
    """
    for paciente_id, detalhes in _SEED_PACIENTE_DETALHES.items():
        conn.execute(
            """
            UPDATE pacientes SET
                sexo = COALESCE(sexo, ?),
                data_nascimento = COALESCE(data_nascimento, ?)
            WHERE id = ?
            """,
            (detalhes["sexo"], detalhes["data_nascimento"], paciente_id),
        )


def _migrar_schema_exames(conn: sqlite3.Connection) -> None:
    """Adiciona as colunas novas de `exames` que ainda nao existirem num banco
    criado por uma versao anterior do schema. Sem isso, um usuario que ja rodou o
    projeto antes desta mudanca ficaria preso no schema antigo (CREATE TABLE IF
    NOT EXISTS nao altera uma tabela que ja existe).
    """
    colunas_existentes = {row[1] for row in conn.execute("PRAGMA table_info(exames)")}
    for coluna, tipo_sql in _EXAME_COLUNAS_NOVAS.items():
        if coluna not in colunas_existentes:
            conn.execute(f"ALTER TABLE exames ADD COLUMN {coluna} {tipo_sql}")


def _seed_exame_detalhes(conn: sqlite3.Connection) -> None:
    """Preenche as colunas novas dos exames semente quando ainda estiverem vazias
    (COALESCE mantem o valor existente se ja houver algum) — roda tanto num banco
    novo (colunas recem-inseridas, tudo NULL) quanto num banco migrado de uma
    versao anterior (colunas recem-adicionadas via ALTER TABLE, tambem NULL).
    """
    for exame_id, detalhes in _SEED_EXAME_DETALHES.items():
        conn.execute(
            """
            UPDATE exames SET
                data_solicitacao = COALESCE(data_solicitacao, ?),
                data_realizacao = COALESCE(data_realizacao, ?),
                resultado = COALESCE(resultado, ?),
                medico_solicitante = COALESCE(medico_solicitante, ?),
                observacoes = COALESCE(observacoes, ?)
            WHERE id = ?
            """,
            (
                detalhes["data_solicitacao"],
                detalhes["data_realizacao"],
                detalhes["resultado"],
                detalhes["medico_solicitante"],
                detalhes["observacoes"],
                exame_id,
            ),
        )


def init_mock_db(force: bool = False) -> None:
    """Cria o banco SQLite com schema e dados sinteticos, se ainda nao existir.
    Tambem migra e preenche os detalhes de exame em bancos ja existentes de uma
    versao anterior do schema (ver _migrar_schema_exames / _seed_exame_detalhes).

    force=True recria do zero (util em testes).
    """
    if force and DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executescript(_SCHEMA)
        _migrar_schema_pacientes(conn)
        _migrar_schema_exames(conn)

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pacientes")
        if cur.fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO pacientes (id, nome_ficticio, historico) VALUES (?, ?, ?)",
                _SEED_PACIENTES,
            )
            conn.executemany(
                "INSERT INTO exames (paciente_id, tipo, status) VALUES (?, ?, ?)",
                _SEED_EXAMES,
            )

        _seed_paciente_detalhes(conn)
        _seed_exame_detalhes(conn)
        conn.commit()


def get_exames_pendentes(paciente_id: int) -> list[str]:
    """Tool chamada pelo no `verificar_exames_pendentes` do LangGraph (agent/nodes.py)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "SELECT tipo FROM exames WHERE paciente_id = ? AND status = 'pendente'",
            (paciente_id,),
        )
        return [row[0] for row in cur.fetchall()]


def get_historico_paciente(paciente_id: int) -> str | None:
    """Tool para contextualizar a resposta da LLM com dados atualizados do paciente."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute("SELECT historico FROM pacientes WHERE id = ?", (paciente_id,))
        row = cur.fetchone()
        return row[0] if row else None


def list_pacientes(nome: str | None = None) -> list[dict]:
    """Lista pacientes do prontuario mock, com filtro opcional por parte do nome
    (case-insensitive — LIKE do SQLite ja ignora caixa para ASCII).

    Usado pelo endpoint GET /patients (ver api/main.py) para o medico localizar o
    ID do paciente antes de chamar POST /assistant/ask.
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        campos = "id, nome_ficticio, historico, sexo, data_nascimento"
        if nome:
            cur = conn.execute(
                f"SELECT {campos} FROM pacientes "
                "WHERE nome_ficticio LIKE ? ORDER BY nome_ficticio",
                (f"%{nome}%",),
            )
        else:
            cur = conn.execute(f"SELECT {campos} FROM pacientes ORDER BY nome_ficticio")
        return [
            {
                "id": row[0],
                "nome_ficticio": row[1],
                "historico": row[2],
                "sexo": row[3],
                "data_nascimento": row[4],
                "idade": _calcular_idade(row[4]),
            }
            for row in cur.fetchall()
        ]


def paciente_existe(paciente_id: int) -> bool:
    """Usado pelos endpoints de exames para diferenciar 'paciente sem exames'
    (lista vazia) de 'paciente nao existe' (404) — ver api/main.py.
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute("SELECT 1 FROM pacientes WHERE id = ?", (paciente_id,))
        return cur.fetchone() is not None


def list_exames_paciente(paciente_id: int) -> list[dict]:
    """Lista TODOS os exames de um paciente (feitos e pendentes), com data de
    solicitacao e de realizacao de cada um (a ultima fica None enquanto o exame
    estiver pendente). Diferente de get_exames_pendentes (que so retorna os tipos
    pendentes, usado internamente pelo no verificar_exames_pendentes do
    LangGraph). Usado pelo endpoint GET /patients/{paciente_id}/exams.
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "SELECT id, tipo, status, data_solicitacao, data_realizacao "
            "FROM exames WHERE paciente_id = ? ORDER BY id",
            (paciente_id,),
        )
        return [
            {
                "id": row[0],
                "tipo": row[1],
                "status": row[2],
                "data_solicitacao": row[3],
                "data_realizacao": row[4],
            }
            for row in cur.fetchall()
        ]


def get_exame(exame_id: int) -> dict | None:
    """Detalhe completo de um exame (todos os campos, incluindo resultado, medico
    solicitante e observacoes). Usado pelo endpoint GET /exams/{exame_id}.
    Retorna None se o exame nao existir.
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "SELECT id, paciente_id, tipo, status, data_solicitacao, data_realizacao, "
            "resultado, medico_solicitante, observacoes FROM exames WHERE id = ?",
            (exame_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "paciente_id": row[1],
            "tipo": row[2],
            "status": row[3],
            "data_solicitacao": row[4],
            "data_realizacao": row[5],
            "resultado": row[6],
            "medico_solicitante": row[7],
            "observacoes": row[8],
        }


if __name__ == "__main__":
    init_mock_db()
    print(f"Banco mock inicializado em {DB_PATH}")
