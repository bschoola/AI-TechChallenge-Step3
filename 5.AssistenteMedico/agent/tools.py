"""Tool de consulta ao 'prontuario' — banco estruturado simulado em SQLite.

Contem tambem o script de inicializacao do banco com dados sinteticos, para que o
projeto rode fim-a-fim sem depender de um sistema hospitalar real.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "prontuarios_mock.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY,
    nome_ficticio TEXT NOT NULL,
    historico TEXT
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL,
    tipo TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pendente', 'concluido')),
    FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);
"""

# Dados de exemplo — todos ficticios, para permitir demonstrar o fluxo no video.
_SEED_PACIENTES = [
    (1, "Paciente Ficticio A", "Historico de hipertensao controlada."),
    (2, "Paciente Ficticio B", "Sem comorbidades registradas."),
]
_SEED_EXAMES = [
    (1, "Mamografia", "pendente"),
    (1, "Hemograma completo", "concluido"),
    (2, "Ultrassom", "pendente"),
]


def init_mock_db(force: bool = False) -> None:
    """Cria o banco SQLite com schema e dados sinteticos, se ainda nao existir.

    force=True recria do zero (util em testes).
    """
    if force and DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executescript(_SCHEMA)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pacientes")
        if cur.fetchone()[0] == 0:
            conn.executemany("INSERT INTO pacientes VALUES (?, ?, ?)", _SEED_PACIENTES)
            conn.executemany(
                "INSERT INTO exames (paciente_id, tipo, status) VALUES (?, ?, ?)",
                _SEED_EXAMES,
            )
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


if __name__ == "__main__":
    init_mock_db()
    print(f"Banco mock inicializado em {DB_PATH}")
