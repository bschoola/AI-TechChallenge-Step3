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

CREATE TABLE IF NOT EXISTS anamnese (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL UNIQUE,
    data_registro TEXT,
    profissional_responsavel TEXT,
    queixa_principal TEXT,
    historia_doenca_atual TEXT,
    historia_patologica_pregressa TEXT,
    historia_familiar TEXT,
    historia_ginecologica_obstetrica TEXT,
    medicamentos_em_uso TEXT,
    alergias TEXT,
    cirurgias_previas TEXT,
    tabagismo TEXT,
    etilismo TEXT,
    atividade_fisica TEXT,
    pressao_arterial TEXT,
    frequencia_cardiaca TEXT,
    frequencia_respiratoria TEXT,
    temperatura_c TEXT,
    saturacao_o2 TEXT,
    peso_kg REAL,
    altura_cm REAL,
    exame_fisico TEXT,
    hipotese_diagnostica TEXT,
    conduta TEXT,
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
    (3, "Paciente Ficticio C", "Quadro de sindrome gripal ha 3 dias, em acompanhamento ambulatorial."),
    (4, "Paciente Ficticio D", "Hipertensao arterial descompensada, em ajuste de medicacao."),
    (5, "Paciente Ficticio E", "Diabetes mellitus tipo 2 descompensado, em investigacao."),
    (6, "Paciente Ficticio F", "Fratura de punho apos queda, em acompanhamento ortopedico."),
    (7, "Paciente Ficticio G", "Quadro de apendicite aguda, encaminhado para avaliacao cirurgica."),
    (8, "Paciente Ficticio H", "Pneumonia adquirida na comunidade, em tratamento antibiotico."),
    (9, "Paciente Ficticio I", "Infeccao do trato urinario em tratamento."),
    (10, "Paciente Ficticio J", "Enxaqueca cronica, em investigacao e ajuste de tratamento."),
    (11, "Paciente Ficticio K", "Colica renal por litiase, em investigacao urologica."),
    (12, "Paciente Ficticio L", "Gastroenterite aguda, com desidratacao leve."),
    (13, "Paciente Ficticio M", "Bronquite aguda, em tratamento sintomatico."),
    (14, "Paciente Ficticio N", "Sindrome respiratoria aguda, investigando COVID-19."),
    (15, "Paciente Ficticio O", "Anemia ferropriva em investigacao, com fadiga associada."),
    (16, "Paciente Ficticio P", "Investigacao de hipotireoidismo, com fadiga e ganho de peso."),
    (17, "Paciente Ficticio Q", "Lombalgia com irradiacao para membro inferior, em investigacao."),
    (18, "Paciente Ficticio R", "Otite media aguda, em tratamento antibiotico."),
    (19, "Paciente Ficticio S", "Conjuntivite bacteriana em tratamento topico."),
    (20, "Paciente Ficticio T", "Suspeita de dengue, em observacao ambulatorial."),
    (21, "Paciente Ficticio U", "Dermatite de contato alergica, em tratamento topico."),
    (22, "Paciente Ficticio V", "Nodulo mamario suspeito em investigacao oncologica."),
    (23, "Paciente Ficticio W", "Cisto mamario benigno, em acompanhamento de rotina."),
    (24, "Paciente Ficticio X", "Cancer de mama em tratamento quimioterapico, em seguimento oncologico."),
    (25, "Paciente Ficticio Y", "Crise asmatica leve a moderada, em tratamento broncodilatador."),
    (26, "Paciente Ficticio Z", "Doenca do refluxo gastroesofagico, em investigacao endoscopica."),
    (27, "Paciente Ficticio AA", "Hernia inguinal a esclarecer, em avaliacao cirurgica."),
    (28, "Paciente Ficticio AB", "Suspeita de trombose venosa profunda em membro inferior."),
    (29, "Paciente Ficticio AC", "Sinusite aguda bacteriana, em tratamento antibiotico."),
    (30, "Paciente Ficticio AD", "Dor cronica generalizada em investigacao, possivel fibromialgia."),
    (31, "Paciente Ficticio AE", "Check-up medico anual de rotina, sem queixas ativas."),
    (32, "Paciente Ficticio AF", "Rastreamento ginecologico e mamario de rotina, sem queixas."),
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
    3: {"sexo": "Masculino", "data_nascimento": "1992-01-01"},
    4: {"sexo": "Feminino", "data_nascimento": "1965-08-14"},
    5: {"sexo": "Masculino", "data_nascimento": "1971-07-27"},
    6: {"sexo": "Feminino", "data_nascimento": "1981-06-13"},
    7: {"sexo": "Masculino", "data_nascimento": "2004-05-26"},
    8: {"sexo": "Feminino", "data_nascimento": "1958-04-12"},
    9: {"sexo": "Feminino", "data_nascimento": "1997-03-25"},
    10: {"sexo": "Feminino", "data_nascimento": "1988-02-11"},
    11: {"sexo": "Masculino", "data_nascimento": "1985-01-24"},
    12: {"sexo": "Feminino", "data_nascimento": "1999-08-10"},
    13: {"sexo": "Masculino", "data_nascimento": "1976-07-23"},
    14: {"sexo": "Feminino", "data_nascimento": "1982-06-09"},
    15: {"sexo": "Feminino", "data_nascimento": "1993-05-22"},
    16: {"sexo": "Feminino", "data_nascimento": "1979-04-08"},
    17: {"sexo": "Masculino", "data_nascimento": "1987-03-21"},
    18: {"sexo": "Masculino", "data_nascimento": "2000-02-07"},
    19: {"sexo": "Feminino", "data_nascimento": "1995-01-20"},
    20: {"sexo": "Masculino", "data_nascimento": "2002-08-06"},
    21: {"sexo": "Feminino", "data_nascimento": "1990-07-19"},
    22: {"sexo": "Feminino", "data_nascimento": "1974-06-05"},
    23: {"sexo": "Feminino", "data_nascimento": "1986-05-18"},
    24: {"sexo": "Feminino", "data_nascimento": "1968-04-04"},
    25: {"sexo": "Masculino", "data_nascimento": "2007-03-17"},
    26: {"sexo": "Masculino", "data_nascimento": "1966-02-03"},
    27: {"sexo": "Masculino", "data_nascimento": "1960-01-16"},
    28: {"sexo": "Feminino", "data_nascimento": "1963-08-02"},
    29: {"sexo": "Feminino", "data_nascimento": "1998-07-15"},
    30: {"sexo": "Feminino", "data_nascimento": "1977-06-01"},
    31: {"sexo": "Masculino", "data_nascimento": "1991-05-14"},
    32: {"sexo": "Feminino", "data_nascimento": "1984-04-27"},
}

# Id explicito (nao mais autoincrement "silencioso") para cada exame semente —
# necessario porque init_mock_db() agora usa "INSERT OR IGNORE" tanto aqui quanto
# em pacientes/anamnese, para conseguir semear pacientes/exames NOVOS num banco
# que o usuario ja rodou antes (sem isso, so o primeiro boot com o banco vazio
# inseria os dados semente — rodar de novo num banco existente nao adicionava
# pacientes/exames novos, so os detalhes/anamnese via UPDATE/INSERT OR IGNORE
# que ja eram idempotentes). A ordem/numeracao aqui bate com as chaves de
# _SEED_EXAME_DETALHES abaixo (1-3 = pacientes 1 e 2 originais, 4-38 = pacientes
# 3-32 novos).
_SEED_EXAMES = [
    (1, 1, "Mamografia", "pendente"),
    (2, 1, "Hemograma completo", "concluido"),
    (3, 2, "Ultrassom", "pendente"),
    (4, 3, "Teste rapido Influenza/COVID-19", "concluido"),
    (5, 4, "Eletrocardiograma", "concluido"),
    (6, 4, "Hemograma completo e perfil lipidico", "pendente"),
    (7, 5, "Hemoglobina glicada (HbA1c)", "pendente"),
    (8, 5, "Glicemia de jejum", "concluido"),
    (9, 6, "Raio-X de punho direito", "concluido"),
    (10, 6, "Reavaliacao ortopedica pos-reducao", "pendente"),
    (11, 7, "Hemograma completo", "concluido"),
    (12, 7, "Ultrassom de abdome", "pendente"),
    (13, 8, "Raio-X de torax", "concluido"),
    (14, 8, "Hemograma completo e PCR", "pendente"),
    (15, 9, "Urocultura com antibiograma", "pendente"),
    (16, 10, "Tomografia de cranio", "pendente"),
    (17, 11, "Tomografia de vias urinarias (sem contraste)", "pendente"),
    (18, 11, "Exame de urina (EAS)", "concluido"),
    (19, 12, "Coprocultura", "pendente"),
    (20, 13, "Raio-X de torax", "pendente"),
    (21, 14, "RT-PCR para SARS-CoV-2", "pendente"),
    (22, 15, "Hemograma completo", "concluido"),
    (23, 15, "Ferritina serica", "pendente"),
    (24, 16, "TSH e T4 livre", "pendente"),
    (25, 17, "Ressonancia magnetica de coluna lombar", "pendente"),
    (26, 20, "Sorologia/antigeno NS1 para dengue", "pendente"),
    (27, 22, "Mamografia bilateral", "concluido"),
    (28, 22, "Biopsia por agulha grossa (core biopsy)", "pendente"),
    (29, 23, "Ultrassom mamario", "concluido"),
    (30, 24, "Hemograma completo (pre-quimioterapia)", "concluido"),
    (31, 24, "Tomografia de torax e abdome (reavaliacao)", "pendente"),
    (32, 26, "Endoscopia digestiva alta", "pendente"),
    (33, 27, "Ultrassom de parede abdominal", "pendente"),
    (34, 28, "Ultrassom Doppler venoso de membros inferiores", "pendente"),
    (35, 30, "Exames laboratoriais de rotina (hemograma, PCR, TSH)", "pendente"),
    (36, 31, "Hemograma e perfil metabolico de rotina", "pendente"),
    (37, 32, "Mamografia de rastreamento", "pendente"),
    (38, 32, "Citologia oncotica (Papanicolau)", "pendente"),
]

# Ficha de anamnese completa dos pacientes semente, por paciente_id — mesma logica
# de _SEED_PACIENTE_DETALHES / _SEED_EXAME_DETALHES: dado ficticio, mas coerente
# com o restante do perfil do paciente (idade, sexo, exame associado).
_SEED_ANAMNESE = {
    1: {  # Paciente Ficticio A — 1972, feminino, rastreio de rotina (mamografia pendente)
        "data_registro": "2026-08-20",
        "profissional_responsavel": "Dra. Camila Nogueira (Ginecologia)",
        "queixa_principal": "Paciente assintomatica, comparece para rastreamento mamario de rotina.",
        "historia_doenca_atual": (
            "Nega nodulos palpaveis, descarga papilar ou dor mamaria. Ultima mamografia "
            "ha mais de 2 anos."
        ),
        "historia_patologica_pregressa": "Hipertensao arterial sistemica, controlada com medicacao.",
        "historia_familiar": (
            "Mae com diagnostico de cancer de mama aos 58 anos. Sem outros casos "
            "oncologicos relatados na familia."
        ),
        "historia_ginecologica_obstetrica": (
            "Menarca aos 12 anos, G2P2A0, menopausa aos 50 anos. Nao faz uso de "
            "terapia de reposicao hormonal."
        ),
        "medicamentos_em_uso": "Losartana 50mg 1x/dia.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Colecistectomia em 2015.",
        "tabagismo": "Nunca fumou.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Caminhada 3x por semana.",
        "pressao_arterial": "128/82 mmHg",
        "frequencia_cardiaca": "76 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.5",
        "saturacao_o2": "98%",
        "peso_kg": 68.5,
        "altura_cm": 162.0,
        "exame_fisico": (
            "Mamas simetricas, sem nodulos palpaveis ou linfonodomegalia axilar. "
            "Sem alteracoes de pele ou complexo areolopapilar."
        ),
        "hipotese_diagnostica": "Rastreamento oncologico de rotina — sem achados suspeitos ao exame fisico.",
        "conduta": "Solicitada mamografia de rastreamento; retorno com resultado em 30 dias.",
    },
    2: {  # Paciente Ficticio B — 1989, feminino, investigacao de nodulo (ultrassom pendente)
        "data_registro": "2026-08-22",
        "profissional_responsavel": "Dr. Paulo Mendes (Radiologia)",
        "queixa_principal": "Nodulo palpavel em mama direita, percebido pela propria paciente ha 3 semanas.",
        "historia_doenca_atual": (
            "Refere nodulo unico, indolor, em quadrante superior lateral da mama "
            "direita, sem crescimento perceptivel desde que notou. Nega descarga "
            "papilar ou alteracoes de pele."
        ),
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar de neoplasias conhecido.",
        "historia_ginecologica_obstetrica": "Menarca aos 13 anos, G0P0A0, ciclos menstruais regulares.",
        "medicamentos_em_uso": "Anticoncepcional oral combinado.",
        "alergias": "Alergia a dipirona (relata urticaria).",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Pratica musculacao 4x por semana.",
        "pressao_arterial": "112/70 mmHg",
        "frequencia_cardiaca": "70 bpm",
        "frequencia_respiratoria": "14 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "99%",
        "peso_kg": 61.0,
        "altura_cm": 167.0,
        "exame_fisico": (
            "Nodulo palpavel de aproximadamente 1,5cm em quadrante superior lateral "
            "da mama direita, movel, contornos regulares. Sem linfonodomegalia axilar "
            "palpavel. Mama esquerda sem alteracoes."
        ),
        "hipotese_diagnostica": "Nodulo mamario a esclarecer — investigacao complementar em andamento.",
        "conduta": "Solicitado ultrassom mamario para caracterizacao do nodulo; retorno com resultado.",
    },
    3: {
        "data_registro": "2026-08-29",
        "profissional_responsavel": "Dra. Beatriz Ramos (Clinica Medica)",
        "queixa_principal": "Febre, dor de garganta, mialgia e coriza ha 3 dias.",
        "historia_doenca_atual": "Inicio subito de febre (aferida em casa, ate 38.8 C), calafrios, dor de garganta, congestao nasal e dores musculares generalizadas. Nega falta de ar.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Paracetamol 750mg se dor ou febre.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Corrida 2x por semana (suspensa durante o quadro atual).",
        "pressao_arterial": "118/76 mmHg",
        "frequencia_cardiaca": "92 bpm",
        "frequencia_respiratoria": "18 irpm",
        "temperatura_c": "38.3",
        "saturacao_o2": "97%",
        "exame_fisico": "Orofaringe hiperemiada sem exsudato purulento. Ausculta pulmonar limpa, sem ruidos adventicios. Sem sinais de desconforto respiratorio.",
        "hipotese_diagnostica": "Sindrome gripal (provavel influenza), sem sinais de gravidade.",
        "conduta": "Tratamento sintomatico, hidratacao e repouso; solicitado teste rapido para influenza/COVID-19 para confirmacao. Retorno se piora ou febre persistente por mais de 5 dias.",
        "peso_kg": 78.0,
        "altura_cm": 175.0,
    },
    4: {
        "data_registro": "2026-08-30",
        "profissional_responsavel": "Dr. Bruno Castro (Cardiologia)",
        "queixa_principal": "Cefaleia occipital e tontura ha 1 semana, pressao alta em casa.",
        "historia_doenca_atual": "Refere aferir pressao em casa com valores frequentemente acima de 160/100 mmHg na ultima semana, associada a cefaleia occipital pela manha e episodios de tontura. Nega dor toracica ou déficit motor.",
        "historia_patologica_pregressa": "Hipertensao arterial sistemica ha 10 anos, dislipidemia.",
        "historia_familiar": "Pai hipertenso, irmao com infarto agudo do miocardio aos 55 anos.",
        "historia_ginecologica_obstetrica": "Menopausa aos 52 anos, G3P3A0.",
        "medicamentos_em_uso": "Losartana 50mg 2x/dia, Sinvastatina 20mg a noite.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Ex-tabagista, parou ha 8 anos.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Sedentaria.",
        "pressao_arterial": "168/102 mmHg",
        "frequencia_cardiaca": "88 bpm",
        "frequencia_respiratoria": "18 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "97%",
        "exame_fisico": "Bulhas cardiacas ritmicas, sem sopros. Sem edema de membros inferiores. Exame neurologico sem déficits focais.",
        "hipotese_diagnostica": "Hipertensao arterial descompensada, provavel ma adesao/dose insuficiente.",
        "conduta": "Ajuste da dose de losartana e associacao de segundo anti-hipertensivo; solicitados exames laboratoriais de rotina e ECG. Retorno em 15 dias com monitoramento domiciliar da pressao.",
        "peso_kg": 82.0,
        "altura_cm": 160.0,
    },
    5: {
        "data_registro": "2026-08-31",
        "profissional_responsavel": "Dra. Juliana Prado (Endocrinologia)",
        "queixa_principal": "Poliuria, polidipsia e perda de peso nao intencional ha 1 mes.",
        "historia_doenca_atual": "Refere sede excessiva, aumento da frequencia urinaria e perda de aproximadamente 4kg no ultimo mes sem mudanca de dieta. Nega visao turva ou feridas de dificil cicatrizacao.",
        "historia_patologica_pregressa": "Diabetes mellitus tipo 2 diagnosticado ha 5 anos, obesidade grau 1.",
        "historia_familiar": "Mae e irma com diabetes tipo 2.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Metformina 850mg 2x/dia.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Sedentario.",
        "pressao_arterial": "134/86 mmHg",
        "frequencia_cardiaca": "80 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.6",
        "saturacao_o2": "98%",
        "exame_fisico": "Mucosas discretamente hipocoradas e desidratadas. Sem lesoes cutaneas ou sinais de neuropatia periferica ao exame simples dos pes.",
        "hipotese_diagnostica": "Diabetes mellitus tipo 2 descompensado — provavel falha terapeutica.",
        "conduta": "Solicitada glicemia de jejum e hemoglobina glicada para reavaliar controle glicemico; mantida metformina e agendado retorno para possivel intensificacao do tratamento.",
        "peso_kg": 94.0,
        "altura_cm": 172.0,
    },
    6: {
        "data_registro": "2026-09-01",
        "profissional_responsavel": "Dr. Eduardo Lima (Ortopedia)",
        "queixa_principal": "Dor e deformidade em punho direito apos queda da propria altura.",
        "historia_doenca_atual": "Refere queda sobre a mao espalmada ha 2 dias, com dor imediata, edema e deformidade visivel em punho direito. Dificuldade para movimentar os dedos.",
        "historia_patologica_pregressa": "Osteopenia diagnosticada em exame de rotina.",
        "historia_familiar": "Mae com osteoporose.",
        "historia_ginecologica_obstetrica": "Menopausa aos 48 anos, G2P2A0.",
        "medicamentos_em_uso": "Carbonato de calcio + vitamina D.",
        "alergias": "Alergia a codeina (relata nauseas intensas).",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Caminhada leve, 2x por semana.",
        "pressao_arterial": "122/78 mmHg",
        "frequencia_cardiaca": "84 bpm",
        "frequencia_respiratoria": "17 irpm",
        "temperatura_c": "36.5",
        "saturacao_o2": "98%",
        "exame_fisico": "Edema e deformidade em dorso de punho direito ('dorso de garfo'), dor a palpacao e a mobilizacao. Perfusao e sensibilidade distal preservadas.",
        "hipotese_diagnostica": "Fratura de radio distal direito (suspeita de fratura de Colles).",
        "conduta": "Imobilizacao provisoria com tala gessada, analgesia e solicitado raio-x de punho para confirmacao; encaminhada para avaliacao ortopedica com o resultado.",
        "peso_kg": 65.0,
        "altura_cm": 165.0,
    },
    7: {
        "data_registro": "2026-09-02",
        "profissional_responsavel": "Dr. Thiago Rocha (Cirurgia Geral)",
        "queixa_principal": "Dor abdominal em fossa iliaca direita ha 18 horas, com febre.",
        "historia_doenca_atual": "Dor iniciada em regiao periumbilical, migrou para fossa iliaca direita, associada a nauseas, um episodio de vomito e febre baixa. Piora com movimentacao.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Musculacao 5x por semana (suspensa pelo quadro atual).",
        "pressao_arterial": "126/80 mmHg",
        "frequencia_cardiaca": "98 bpm",
        "frequencia_respiratoria": "19 irpm",
        "temperatura_c": "37.9",
        "saturacao_o2": "98%",
        "exame_fisico": "Abdome doloroso a palpacao em fossa iliaca direita, com sinal de Blumberg positivo. Ruidos hidroaereos diminuidos.",
        "hipotese_diagnostica": "Apendicite aguda — indicacao de avaliacao cirurgica de urgencia.",
        "conduta": "Solicitados hemograma e ultrassom de abdome para confirmacao diagnostica; encaminhado para avaliacao da cirurgia geral com possivel indicacao de apendicectomia.",
        "peso_kg": 70.0,
        "altura_cm": 178.0,
    },
    8: {
        "data_registro": "2026-09-03",
        "profissional_responsavel": "Dra. Fernanda Alves (Pneumologia)",
        "queixa_principal": "Tosse produtiva, febre e falta de ar progressiva ha 4 dias.",
        "historia_doenca_atual": "Tosse com expectoracao amarelada, febre ate 38.9 C, dor toracica ventilatorio-dependente e dispneia aos moderados esforcos, com piora progressiva nos ultimos 2 dias.",
        "historia_patologica_pregressa": "Hipertensao arterial sistemica controlada.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Menopausa aos 51 anos, G4P4A0.",
        "medicamentos_em_uso": "Enalapril 10mg 1x/dia.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Ex-tabagista, parou ha 15 anos (carga tabagica de 20 anos-maco).",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Sedentaria.",
        "pressao_arterial": "130/84 mmHg",
        "frequencia_cardiaca": "102 bpm",
        "frequencia_respiratoria": "24 irpm",
        "temperatura_c": "38.6",
        "saturacao_o2": "93%",
        "exame_fisico": "Estertores crepitantes em base pulmonar direita a ausculta, com diminuicao do murmurio vesicular localizado. Taquipneica em repouso.",
        "hipotese_diagnostica": "Pneumonia adquirida na comunidade, base pulmonar direita.",
        "conduta": "Iniciada antibioticoterapia empirica, solicitado raio-x de torax e monitorada saturacao de oxigenio; reavaliacao em 48-72h ou antes se piora do quadro respiratorio.",
        "peso_kg": 71.0,
        "altura_cm": 158.0,
    },
    9: {
        "data_registro": "2026-09-04",
        "profissional_responsavel": "Dr. Marcos Vieira (Urologia)",
        "queixa_principal": "Ardencia miccional e aumento da frequencia urinaria ha 2 dias.",
        "historia_doenca_atual": "Refere disuria, polaciuria e desconforto suprapubico ha 2 dias, sem febre ou dor lombar. Nega corrimento vaginal ou hematuria.",
        "historia_patologica_pregressa": "Episodios previos de infeccao urinaria (2 nos ultimos 12 meses).",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G0P0A0, uso de anticoncepcional oral.",
        "medicamentos_em_uso": "Anticoncepcional oral combinado.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Yoga 2x por semana.",
        "pressao_arterial": "114/72 mmHg",
        "frequencia_cardiaca": "78 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.8",
        "saturacao_o2": "99%",
        "exame_fisico": "Dor a palpacao suprapubica leve. Punho-percussao lombar (Giordano) negativa bilateralmente. Sem sinais de toxemia.",
        "hipotese_diagnostica": "Infeccao do trato urinario baixo (cistite), provavelmente recorrente.",
        "conduta": "Iniciado antibiotico empirico e solicitada urocultura com antibiograma para orientar tratamento definitivo, dada a recorrencia dos episodios.",
        "peso_kg": 60.0,
        "altura_cm": 163.0,
    },
    10: {
        "data_registro": "2026-08-29",
        "profissional_responsavel": "Dr. Andre Barbosa (Neurologia)",
        "queixa_principal": "Cefaleia pulsatil recorrente, ha mais de 15 dias por mes.",
        "historia_doenca_atual": "Refere cefaleia pulsatil unilateral, associada a nausea e fotofobia, com frequencia de mais de 15 dias por mes nos ultimos 3 meses, impactando o trabalho.",
        "historia_patologica_pregressa": "Enxaqueca desde a adolescencia.",
        "historia_familiar": "Mae com historico de enxaqueca.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, piora da cefaleia no periodo perimenstrual.",
        "medicamentos_em_uso": "Sumatriptano se dor (uso frequente, ~10x por mes).",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Sedentaria (cefaleia limita a pratica regular).",
        "pressao_arterial": "116/74 mmHg",
        "frequencia_cardiaca": "72 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "99%",
        "exame_fisico": "Exame neurologico sem déficits focais. Sem sinais de irritacao meningea ou papiledema a fundoscopia simples.",
        "hipotese_diagnostica": "Enxaqueca cronica com provavel uso excessivo de medicacao sintomatica.",
        "conduta": "Iniciada profilaxia medicamentosa e orientacao para reducao do uso de triptano; solicitada tomografia de cranio para excluir causas secundarias dado o padrao de piora recente.",
        "peso_kg": 63.0,
        "altura_cm": 167.0,
    },
    11: {
        "data_registro": "2026-08-30",
        "profissional_responsavel": "Dr. Marcos Vieira (Urologia)",
        "queixa_principal": "Dor lombar em colica, irradiando para região inguinal, ha 6 horas.",
        "historia_doenca_atual": "Dor lombar direita de forte intensidade, tipo colica, irradiando para regiao inguinal, associada a nauseas e um episodio de hematuria macroscopica.",
        "historia_patologica_pregressa": "Episodio previo de calculo renal ha 3 anos, eliminado espontaneamente.",
        "historia_familiar": "Pai com historico de calculos renais.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Alergia a dipirona (relata rash cutaneo).",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Ciclismo aos finais de semana.",
        "pressao_arterial": "138/88 mmHg",
        "frequencia_cardiaca": "96 bpm",
        "frequencia_respiratoria": "18 irpm",
        "temperatura_c": "36.9",
        "saturacao_o2": "98%",
        "exame_fisico": "Punho-percussao lombar (Giordano) positiva a direita. Abdome flacido, sem sinais de irritacao peritoneal.",
        "hipotese_diagnostica": "Colica renal por provavel litiase ureteral direita.",
        "conduta": "Analgesia e hidratacao venosa; solicitada tomografia de vias urinarias sem contraste para localizar e dimensionar o calculo.",
        "peso_kg": 85.0,
        "altura_cm": 180.0,
    },
    12: {
        "data_registro": "2026-08-31",
        "profissional_responsavel": "Dra. Beatriz Ramos (Clinica Medica)",
        "queixa_principal": "Diarreia e vomitos ha 2 dias, apos refeicao fora de casa.",
        "historia_doenca_atual": "Iniciou diarreia aquosa (cerca de 6 episodios/dia) e vomitos apos refeicao em restaurante ha 2 dias, associada a dor abdominal em colica e febre baixa.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G0P0A0.",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Pilates 3x por semana (suspensa pelo quadro atual).",
        "pressao_arterial": "106/68 mmHg",
        "frequencia_cardiaca": "94 bpm",
        "frequencia_respiratoria": "18 irpm",
        "temperatura_c": "37.6",
        "saturacao_o2": "98%",
        "exame_fisico": "Mucosas discretamente secas. Abdome doloroso difusamente a palpacao, sem sinais de irritacao peritoneal. Ruidos hidroaereos aumentados.",
        "hipotese_diagnostica": "Gastroenterite aguda, provavel origem infecciosa alimentar, com desidratacao leve.",
        "conduta": "Hidratacao oral/venosa conforme tolerancia e dieta leve; solicitada coprocultura caso os sintomas persistam alem de 5 dias. Retorno se sinais de alarme (sangue nas fezes, febre alta persistente).",
        "peso_kg": 58.0,
        "altura_cm": 162.0,
    },
    13: {
        "data_registro": "2026-09-01",
        "profissional_responsavel": "Dra. Fernanda Alves (Pneumologia)",
        "queixa_principal": "Tosse seca persistente ha 10 dias, com piora noturna.",
        "historia_doenca_atual": "Tosse inicialmente seca, ha 10 dias, evoluindo com pequena quantidade de expectoracao esbranquicada, sem febre. Refere desconforto retroesternal ao tossir.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Fumante ativo, 15 cigarros/dia ha 20 anos.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Sedentario.",
        "pressao_arterial": "128/82 mmHg",
        "frequencia_cardiaca": "86 bpm",
        "frequencia_respiratoria": "19 irpm",
        "temperatura_c": "36.9",
        "saturacao_o2": "96%",
        "exame_fisico": "Ausculta pulmonar com roncos esparsos bilaterais, sem estertores crepitantes. Sem sinais de desconforto respiratorio em repouso.",
        "hipotese_diagnostica": "Bronquite aguda, provavelmente de origem viral.",
        "conduta": "Tratamento sintomatico e orientacao para cessacao do tabagismo; solicitado raio-x de torax pela duracao prolongada da tosse e carga tabagica.",
        "peso_kg": 88.0,
        "altura_cm": 174.0,
    },
    14: {
        "data_registro": "2026-09-02",
        "profissional_responsavel": "Dra. Patricia Souza (Infectologia)",
        "queixa_principal": "Febre, perda de olfato e tosse seca ha 3 dias.",
        "historia_doenca_atual": "Refere febre (ate 38.2 C), anosmia, tosse seca e cansaco ha 3 dias. Contato domiciliar recente com caso confirmado de COVID-19. Nega falta de ar.",
        "historia_patologica_pregressa": "Asma leve intermitente.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G1P1A0.",
        "medicamentos_em_uso": "Salbutamol spray se necessario.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Caminhada 3x por semana (suspensa pelo quadro atual).",
        "pressao_arterial": "118/76 mmHg",
        "frequencia_cardiaca": "90 bpm",
        "frequencia_respiratoria": "18 irpm",
        "temperatura_c": "38.0",
        "saturacao_o2": "97%",
        "exame_fisico": "Ausculta pulmonar limpa, sem ruidos adventicios. Sem sinais de desconforto respiratorio. Orofaringe sem alteracoes.",
        "hipotese_diagnostica": "Sindrome respiratoria aguda, provavel COVID-19 (contato domiciliar confirmado).",
        "conduta": "Isolamento domiciliar, tratamento sintomatico e solicitado teste RT-PCR para confirmacao; orientada a monitorar saturacao e retornar se dispneia.",
        "peso_kg": 69.0,
        "altura_cm": 165.0,
    },
    15: {
        "data_registro": "2026-09-03",
        "profissional_responsavel": "Dra. Isabela Teixeira (Hematologia)",
        "queixa_principal": "Fadiga intensa e palidez percebida ha cerca de 1 mes.",
        "historia_doenca_atual": "Refere cansaco desproporcional as atividades habituais, palidez cutanea notada por familiares e episodios de tontura ao levantar. Menstruacoes volumosas nos ultimos ciclos.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Ciclos menstruais irregulares e volumosos (menorragia), G1P1A0.",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Sedentaria (fadiga limita a pratica).",
        "pressao_arterial": "108/70 mmHg",
        "frequencia_cardiaca": "98 bpm",
        "frequencia_respiratoria": "17 irpm",
        "temperatura_c": "36.5",
        "saturacao_o2": "98%",
        "exame_fisico": "Mucosas e conjuntivas hipocoradas (++/4+). Sem linfonodomegalias ou hepatoesplenomegalia palpavel.",
        "hipotese_diagnostica": "Anemia ferropriva, provavelmente secundaria a menorragia.",
        "conduta": "Solicitados hemograma completo e ferritina para confirmar e quantificar a anemia; encaminhamento a ginecologia para investigar a menorragia.",
        "peso_kg": 56.0,
        "altura_cm": 160.0,
    },
    16: {
        "data_registro": "2026-09-04",
        "profissional_responsavel": "Dra. Juliana Prado (Endocrinologia)",
        "queixa_principal": "Fadiga, ganho de peso e intolerancia ao frio ha 3 meses.",
        "historia_doenca_atual": "Refere cansaco progressivo, ganho de aproximadamente 5kg sem mudanca de habitos, pele seca, queda de cabelo e maior sensibilidade ao frio nos ultimos 3 meses.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Mae com hipotireoidismo (tireoidite de Hashimoto).",
        "historia_ginecologica_obstetrica": "Ciclos menstruais irregulares nos ultimos meses, G2P2A0.",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Sedentaria.",
        "pressao_arterial": "116/76 mmHg",
        "frequencia_cardiaca": "62 bpm",
        "frequencia_respiratoria": "15 irpm",
        "temperatura_c": "36.1",
        "saturacao_o2": "98%",
        "exame_fisico": "Pele seca e fria ao toque, discreto edema periorbital. Tireoide discretamente aumentada a palpacao, sem nodulos evidentes.",
        "hipotese_diagnostica": "Hipotireoidismo primario a esclarecer, possivel tireoidite autoimune.",
        "conduta": "Solicitados TSH, T4 livre e anti-TPO para investigacao etiologica; retorno com resultados para eventual inicio de reposicao hormonal.",
        "peso_kg": 74.0,
        "altura_cm": 164.0,
    },
    17: {
        "data_registro": "2026-08-29",
        "profissional_responsavel": "Dra. Marina Cunha (Reumatologia)",
        "queixa_principal": "Dor lombar irradiando para perna esquerda ha 3 semanas.",
        "historia_doenca_atual": "Dor lombar baixa iniciada apos esforco fisico (levantamento de peso), com irradiacao para face posterior da coxa e perna esquerda, piora ao sentar e tossir. Refere formigamento ocasional no pe.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Pai com hernia de disco lombar.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Anti-inflamatorio nao esteroidal se dor.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Musculacao 4x por semana (suspensa pelo quadro atual).",
        "pressao_arterial": "130/84 mmHg",
        "frequencia_cardiaca": "80 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.5",
        "saturacao_o2": "98%",
        "exame_fisico": "Manobra de Lasegue positiva a esquerda. Forca muscular preservada, reflexos patelares e aquileus simetricos. Dor a palpacao paravertebral lombar.",
        "hipotese_diagnostica": "Lombociatalgia esquerda, suspeita de hernia de disco lombar.",
        "conduta": "Analgesia, anti-inflamatorio e encaminhamento para fisioterapia; solicitada ressonancia magnetica de coluna lombar para confirmacao diagnostica.",
        "peso_kg": 90.0,
        "altura_cm": 176.0,
    },
    18: {
        "data_registro": "2026-08-30",
        "profissional_responsavel": "Dr. Vinicius Duarte (Otorrinolaringologia)",
        "queixa_principal": "Dor de ouvido direito e sensacao de ouvido tampado ha 2 dias.",
        "historia_doenca_atual": "Refere otalgia direita de forte intensidade, associada a hipoacusia e febre baixa, apos resfriado comum na semana anterior. Nega secrecao purulenta no momento.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Futebol 1x por semana.",
        "pressao_arterial": "120/78 mmHg",
        "frequencia_cardiaca": "84 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "37.5",
        "saturacao_o2": "99%",
        "exame_fisico": "Otoscopia direita com membrana timpanica hiperemiada e abaulada. Otoscopia esquerda sem alteracoes.",
        "hipotese_diagnostica": "Otite media aguda a direita.",
        "conduta": "Iniciado antibiotico oral e analgesia; retorno em 7 dias ou antes se piora ou secrecao purulenta.",
        "peso_kg": 76.0,
        "altura_cm": 177.0,
    },
    19: {
        "data_registro": "2026-08-31",
        "profissional_responsavel": "Dra. Carla Nunes (Oftalmologia)",
        "queixa_principal": "Olho direito vermelho e com secrecao purulenta ha 2 dias.",
        "historia_doenca_atual": "Refere hiperemia conjuntival, secrecao purulenta amarelada e sensacao de corpo estranho no olho direito ha 2 dias, com piora ao acordar (cilios colados).",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G0P0A0.",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Natacao 2x por semana (suspensa pelo quadro atual).",
        "pressao_arterial": "112/72 mmHg",
        "frequencia_cardiaca": "76 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.5",
        "saturacao_o2": "99%",
        "exame_fisico": "Hiperemia conjuntival difusa em olho direito, com secrecao purulenta presente. Cornea transparente, sem lesoes aparentes. Olho esquerdo sem alteracoes.",
        "hipotese_diagnostica": "Conjuntivite bacteriana em olho direito.",
        "conduta": "Prescrito colirio antibiotico topico e orientacoes de higiene ocular; retorno se nao houver melhora em 3-5 dias.",
        "peso_kg": 59.0,
        "altura_cm": 164.0,
    },
    20: {
        "data_registro": "2026-09-01",
        "profissional_responsavel": "Dra. Patricia Souza (Infectologia)",
        "queixa_principal": "Febre alta, dor atras dos olhos e dores no corpo ha 3 dias.",
        "historia_doenca_atual": "Febre alta (ate 39.5 C), cefaleia retro-orbitaria, mialgia intensa e algumas manchas avermelhadas no tronco ha 3 dias. Regiao com casos confirmados de dengue na vizinhanca.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Paracetamol se febre (evitando anti-inflamatorios).",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Corrida 3x por semana (suspensa pelo quadro atual).",
        "pressao_arterial": "110/70 mmHg",
        "frequencia_cardiaca": "92 bpm",
        "frequencia_respiratoria": "18 irpm",
        "temperatura_c": "39.2",
        "saturacao_o2": "98%",
        "exame_fisico": "Exantema maculopapular discreto em tronco. Prova do laco nao realizada nesta consulta. Sem sinais de sangramento ativo ou sinais de alarme no momento.",
        "hipotese_diagnostica": "Sindrome febril aguda, suspeita de dengue.",
        "conduta": "Hidratacao oral abundante, antitermico e orientacao sobre sinais de alarme; solicitada sorologia/antigeno NS1 para confirmacao e retorno diario para reavaliacao clinica.",
        "peso_kg": 72.0,
        "altura_cm": 173.0,
    },
    21: {
        "data_registro": "2026-09-02",
        "profissional_responsavel": "Dra. Larissa Martins (Dermatologia)",
        "queixa_principal": "Lesoes avermelhadas e com coceira nos bracos ha 5 dias.",
        "historia_doenca_atual": "Refere surgimento de lesoes eritematosas, pruriginosas e com pequenas vesiculas nos antebracos ha 5 dias, apos uso de um novo produto de limpeza domestica.",
        "historia_patologica_pregressa": "Rinite alergica.",
        "historia_familiar": "Mae com historico de dermatite atopica.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G1P1A0.",
        "medicamentos_em_uso": "Loratadina 10mg se necessario.",
        "alergias": "Alergia a niquel (relata reacao a bijuterias).",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Pilates 2x por semana.",
        "pressao_arterial": "114/74 mmHg",
        "frequencia_cardiaca": "74 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "99%",
        "exame_fisico": "Placas eritematosas com vesiculas puntiformes em antebracos bilaterais, bem delimitadas, sem sinais de infeccao secundaria.",
        "hipotese_diagnostica": "Dermatite de contato alergica, provavel relacao com produto de limpeza.",
        "conduta": "Suspensao do produto suspeito, corticoide topico e anti-histaminico oral; retorno se nao houver melhora em 1 semana ou piora das lesoes.",
        "peso_kg": 66.0,
        "altura_cm": 168.0,
    },
    22: {
        "data_registro": "2026-09-03",
        "profissional_responsavel": "Dra. Renata Farias (Oncologia)",
        "queixa_principal": "Nodulo endurecido em mama esquerda, percebido ha 1 mes.",
        "historia_doenca_atual": "Refere nodulo unico, endurecido, indolor, em quadrante superior lateral da mama esquerda, com discreto crescimento percebido nas ultimas semanas. Nega descarga papilar.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Irma com cancer de mama diagnosticado aos 55 anos.",
        "historia_ginecologica_obstetrica": "Menopausa aos 50 anos, G2P2A0, nao usa terapia de reposicao hormonal.",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Ex-tabagista, parou ha 10 anos.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Caminhada 2x por semana.",
        "pressao_arterial": "124/80 mmHg",
        "frequencia_cardiaca": "78 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.5",
        "saturacao_o2": "98%",
        "exame_fisico": "Nodulo endurecido, de bordas irregulares, de aproximadamente 2cm em quadrante superior lateral da mama esquerda, pouco movel. Linfonodo axilar esquerdo palpavel, aumentado.",
        "hipotese_diagnostica": "Nodulo mamario suspeito — investigacao para exclusao de neoplasia maligna.",
        "conduta": "Solicitadas mamografia bilateral e biopsia por agulha grossa do nodulo para definicao diagnostica; encaminhada com prioridade dado o historico familiar e os achados ao exame.",
        "peso_kg": 70.0,
        "altura_cm": 161.0,
    },
    23: {
        "data_registro": "2026-09-04",
        "profissional_responsavel": "Dr. Paulo Mendes (Radiologia)",
        "queixa_principal": "Nodulo mamario doloroso, de aparecimento ciclico, ha 2 meses.",
        "historia_doenca_atual": "Refere nodulo palpavel em mama direita, com dor que se intensifica no periodo pre-menstrual e melhora apos a menstruacao, sugerindo natureza ciclica/hormonal.",
        "historia_patologica_pregressa": "Mastalgia ciclica ha varios anos.",
        "historia_familiar": "Sem historico familiar de neoplasias conhecido.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G2P2A0.",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Corrida 2x por semana.",
        "pressao_arterial": "118/76 mmHg",
        "frequencia_cardiaca": "72 bpm",
        "frequencia_respiratoria": "15 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "99%",
        "exame_fisico": "Nodulo amolecido, movel, de contornos regulares, em mama direita, compativel com formacao cistica. Sem linfonodomegalia axilar.",
        "hipotese_diagnostica": "Cisto mamario simples, achado benigno provavel.",
        "conduta": "Solicitado ultrassom mamario para caracterizacao; conduta expectante se confirmada natureza cistica simples, com retorno anual de rotina.",
        "peso_kg": 64.0,
        "altura_cm": 166.0,
    },
    24: {
        "data_registro": "2026-08-29",
        "profissional_responsavel": "Dra. Renata Farias (Oncologia)",
        "queixa_principal": "Retorno de seguimento oncologico, 3º ciclo de quimioterapia.",
        "historia_doenca_atual": "Paciente em tratamento de carcinoma ductal invasivo de mama direita, diagnosticado ha 4 meses, atualmente no 3º ciclo de quimioterapia neoadjuvante. Refere fadiga e nauseas leves apos as sessoes, sem febre.",
        "historia_patologica_pregressa": "Carcinoma ductal invasivo de mama direita (estadio II), hipertensao arterial controlada.",
        "historia_familiar": "Mae e irma com cancer de mama.",
        "historia_ginecologica_obstetrica": "Menopausa aos 49 anos, G3P3A0.",
        "medicamentos_em_uso": "Protocolo quimioterapico institucional, Losartana 50mg 1x/dia, antiemetico se nausea.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Biopsia por agulha grossa de mama direita ha 4 meses.",
        "tabagismo": "Nunca fumou.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Caminhada leve conforme tolerancia durante o tratamento.",
        "pressao_arterial": "126/80 mmHg",
        "frequencia_cardiaca": "88 bpm",
        "frequencia_respiratoria": "17 irpm",
        "temperatura_c": "36.6",
        "saturacao_o2": "97%",
        "exame_fisico": "Reducao perceptivel do tumor palpavel em mama direita em relacao a avaliacao anterior. Alopecia relacionada a quimioterapia. Sem sinais de infeccao ou neutropenia febril.",
        "hipotese_diagnostica": "Carcinoma ductal invasivo de mama direita, em resposta parcial ao tratamento neoadjuvante.",
        "conduta": "Mantido protocolo quimioterapico conforme planejado; solicitados hemograma de controle pre-quimioterapia e tomografia de reavaliacao apos o 4º ciclo.",
        "peso_kg": 67.0,
        "altura_cm": 159.0,
    },
    25: {
        "data_registro": "2026-08-30",
        "profissional_responsavel": "Dra. Fernanda Alves (Pneumologia)",
        "queixa_principal": "Falta de ar e chiado no peito ha 1 dia, apos exposicao a poeira.",
        "historia_doenca_atual": "Refere dispneia e sibilancia apos limpeza de um deposito empoeirado, com necessidade de uso frequente de bombinha de resgate nas ultimas horas.",
        "historia_patologica_pregressa": "Asma bronquica desde a infancia.",
        "historia_familiar": "Mae com asma, pai com rinite alergica.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Salbutamol spray de resgate, budesonida inalatoria de manutencao.",
        "alergias": "Alergia a acaros (rinite alergica associada).",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Futebol 2x por semana (evitado durante crises).",
        "pressao_arterial": "122/78 mmHg",
        "frequencia_cardiaca": "100 bpm",
        "frequencia_respiratoria": "24 irpm",
        "temperatura_c": "36.6",
        "saturacao_o2": "95%",
        "exame_fisico": "Sibilos expiratorios difusos a ausculta pulmonar bilateral, uso discreto de musculatura acessoria. Fala frases completas.",
        "hipotese_diagnostica": "Crise asmatica leve a moderada, desencadeada por exposicao a alergeno.",
        "conduta": "Broncodilatador inalatorio em consultorio com boa resposta, corticoide oral por curto periodo e reforco do plano de acao para asma; retorno se piora ou nao melhora em 24-48h.",
        "peso_kg": 68.0,
        "altura_cm": 172.0,
    },
    26: {
        "data_registro": "2026-08-31",
        "profissional_responsavel": "Dr. Diego Almeida (Gastroenterologia)",
        "queixa_principal": "Queimacao retroesternal e regurgitacao apos as refeicoes ha 2 meses.",
        "historia_doenca_atual": "Refere pirose e regurgitacao acida frequentes, principalmente apos refeicoes copiosas e ao deitar-se, com piora progressiva nos ultimos 2 meses.",
        "historia_patologica_pregressa": "Obesidade grau 1, hipertensao arterial controlada.",
        "historia_familiar": "Pai com hernia de hiato.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Losartana 50mg 1x/dia, omeprazol 20mg conforme necessidade.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Ex-tabagista, parou ha 5 anos.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Sedentario.",
        "pressao_arterial": "132/84 mmHg",
        "frequencia_cardiaca": "78 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "97%",
        "exame_fisico": "Abdome globoso, indolor a palpacao, sem massas ou visceromegalias palpaveis. Sem sinais de alarme (disfagia, emagrecimento, sangramento).",
        "hipotese_diagnostica": "Doenca do refluxo gastroesofagico, sem sinais de alarme no momento.",
        "conduta": "Otimizacao do inibidor de bomba de protons e orientacoes dieteticas/posturais; solicitada endoscopia digestiva alta pela cronicidade dos sintomas.",
        "peso_kg": 91.0,
        "altura_cm": 170.0,
    },
    27: {
        "data_registro": "2026-09-01",
        "profissional_responsavel": "Dr. Thiago Rocha (Cirurgia Geral)",
        "queixa_principal": "Abaulamento em regiao inguinal direita, que aumenta ao esforco.",
        "historia_doenca_atual": "Percebeu abaulamento na regiao inguinal direita ha cerca de 2 meses, que aumenta ao tossir ou fazer esforco fisico e reduz espontaneamente ao deitar. Refere desconforto leve, sem dor intensa ou nauseas.",
        "historia_patologica_pregressa": "Hiperplasia prostatica benigna.",
        "historia_familiar": "Pai operado de hernia inguinal.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Tansulosina 0.4mg a noite.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Ex-tabagista, parou ha 12 anos.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Jardinagem regular (esforco fisico moderado).",
        "pressao_arterial": "136/86 mmHg",
        "frequencia_cardiaca": "76 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.5",
        "saturacao_o2": "97%",
        "exame_fisico": "Abaulamento redutivel em regiao inguinal direita, que se exacerba a manobra de Valsalva. Sem sinais de encarceramento ou irritacao peritoneal.",
        "hipotese_diagnostica": "Hernia inguinal direita, nao complicada no momento.",
        "conduta": "Orientado sobre sinais de alerta de encarceramento/estrangulamento; encaminhado para avaliacao cirurgica eletiva com possivel indicacao de correcao herniaria.",
        "peso_kg": 79.0,
        "altura_cm": 169.0,
    },
    28: {
        "data_registro": "2026-09-02",
        "profissional_responsavel": "Dr. Gustavo Pires (Cirurgia Vascular)",
        "queixa_principal": "Edema e dor na panturrilha esquerda ha 3 dias.",
        "historia_doenca_atual": "Refere edema unilateral, dor e sensacao de peso na panturrilha esquerda ha 3 dias, apos viagem de longa duracao de aviao. Nega falta de ar ou dor toracica.",
        "historia_patologica_pregressa": "Hipertensao arterial controlada, varizes de membros inferiores.",
        "historia_familiar": "Mae com historico de trombose venosa.",
        "historia_ginecologica_obstetrica": "Menopausa aos 53 anos, G2P2A0.",
        "medicamentos_em_uso": "Anlodipino 5mg 1x/dia.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Ex-tabagista, parou ha 20 anos.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Sedentaria.",
        "pressao_arterial": "140/88 mmHg",
        "frequencia_cardiaca": "84 bpm",
        "frequencia_respiratoria": "17 irpm",
        "temperatura_c": "36.7",
        "saturacao_o2": "97%",
        "exame_fisico": "Edema assimetrico em panturrilha esquerda, com aumento de temperatura local e dor a palpacao do trajeto venoso profundo. Sinal de Homans duvidoso.",
        "hipotese_diagnostica": "Suspeita de trombose venosa profunda em membro inferior esquerdo.",
        "conduta": "Encaminhada com prioridade para ultrassom Doppler venoso de membros inferiores; orientada repouso relativo com elevacao do membro ate o resultado, dado o risco de embolia pulmonar.",
        "peso_kg": 77.0,
        "altura_cm": 158.0,
    },
    29: {
        "data_registro": "2026-09-03",
        "profissional_responsavel": "Dr. Vinicius Duarte (Otorrinolaringologia)",
        "queixa_principal": "Dor facial, congestao nasal e secrecao purulenta ha 10 dias.",
        "historia_doenca_atual": "Refere quadro gripal ha 10 dias com piora nos ultimos 3 dias: dor em regiao maxilar e frontal, secrecao nasal purulenta e febre baixa, sugerindo sobreposicao bacteriana.",
        "historia_patologica_pregressa": "Rinite alergica.",
        "historia_familiar": "Sem historico familiar relevante.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G0P0A0.",
        "medicamentos_em_uso": "Loratadina 10mg se necessario.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Danca 2x por semana (suspensa pelo quadro atual).",
        "pressao_arterial": "110/70 mmHg",
        "frequencia_cardiaca": "82 bpm",
        "frequencia_respiratoria": "17 irpm",
        "temperatura_c": "37.7",
        "saturacao_o2": "98%",
        "exame_fisico": "Dor a palpacao de seios maxilares e frontais bilateralmente. Secrecao purulenta em meato nasal medio a rinoscopia anterior.",
        "hipotese_diagnostica": "Sinusite aguda bacteriana, provavel complicacao de quadro viral previo.",
        "conduta": "Iniciado antibiotico oral, lavagem nasal com solucao salina e analgesia; retorno em 10 dias ou antes se piora dos sintomas.",
        "peso_kg": 61.0,
        "altura_cm": 165.0,
    },
    30: {
        "data_registro": "2026-09-04",
        "profissional_responsavel": "Dra. Marina Cunha (Reumatologia)",
        "queixa_principal": "Dor generalizada no corpo e fadiga persistente ha mais de 6 meses.",
        "historia_doenca_atual": "Refere dor musculoesqueletica difusa, ha mais de 6 meses, associada a fadiga importante, sono nao reparador e dificuldade de concentracao. Sintomas pioram com estresse e frio.",
        "historia_patologica_pregressa": "Enxaqueca ocasional, sindrome do intestino irritavel.",
        "historia_familiar": "Mae com diagnostico de fibromialgia.",
        "historia_ginecologica_obstetrica": "Ciclos regulares, G1P1A0.",
        "medicamentos_em_uso": "Nenhum de uso continuo no momento.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Sedentaria (dor limita a pratica regular).",
        "pressao_arterial": "118/78 mmHg",
        "frequencia_cardiaca": "80 bpm",
        "frequencia_respiratoria": "16 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "98%",
        "exame_fisico": "Dor a palpacao de multiplos pontos musculoesqueleticos, simetricos, sem sinais de sinovite ou edema articular. Forca muscular preservada.",
        "hipotese_diagnostica": "Dor cronica generalizada, quadro compativel com fibromialgia.",
        "conduta": "Solicitados exames laboratoriais de rotina para excluir causas secundarias (inflamatorias/reumatologicas); orientada atividade fisica gradual e encaminhada para acompanhamento multidisciplinar.",
        "peso_kg": 68.0,
        "altura_cm": 163.0,
    },
    31: {
        "data_registro": "2026-08-29",
        "profissional_responsavel": "Dra. Beatriz Ramos (Clinica Medica)",
        "queixa_principal": "Paciente assintomatico, comparece para check-up anual de rotina.",
        "historia_doenca_atual": "Nega queixas atuais. Comparece para avaliacao clinica e exames de rotina anuais.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Pai hipertenso; sem outros antecedentes relevantes.",
        "historia_ginecologica_obstetrica": "Nao aplicavel (paciente do sexo masculino).",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Nenhuma.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Consumo social, ocasional.",
        "atividade_fisica": "Musculacao e corrida, 4x por semana.",
        "pressao_arterial": "118/76 mmHg",
        "frequencia_cardiaca": "64 bpm",
        "frequencia_respiratoria": "14 irpm",
        "temperatura_c": "36.4",
        "saturacao_o2": "99%",
        "exame_fisico": "Exame fisico geral sem alteracoes. Ausculta cardiaca e pulmonar normais.",
        "hipotese_diagnostica": "Paciente higido — sem achados relevantes ao exame clinico de rotina.",
        "conduta": "Solicitados exames laboratoriais de rotina (check-up anual); mantidas orientacoes de estilo de vida saudavel e retorno em 12 meses.",
        "peso_kg": 80.0,
        "altura_cm": 179.0,
    },
    32: {
        "data_registro": "2026-08-30",
        "profissional_responsavel": "Dra. Camila Nogueira (Ginecologia)",
        "queixa_principal": "Paciente assintomatica, comparece para rastreamento ginecologico anual.",
        "historia_doenca_atual": "Nega nodulos mamarios, sangramento anormal ou outras queixas. Ultimo Papanicolau e mamografia ha aproximadamente 1 ano, ambos normais.",
        "historia_patologica_pregressa": "Sem comorbidades registradas.",
        "historia_familiar": "Sem historico familiar de neoplasias conhecido.",
        "historia_ginecologica_obstetrica": "Menarca aos 12 anos, G1P1A0, ciclos menstruais regulares.",
        "medicamentos_em_uso": "Nenhum de uso continuo.",
        "alergias": "Nega alergias medicamentosas conhecidas.",
        "cirurgias_previas": "Cesariana ha 10 anos.",
        "tabagismo": "Nao fumante.",
        "etilismo": "Nega uso de bebida alcoolica.",
        "atividade_fisica": "Yoga e caminhada, 3x por semana.",
        "pressao_arterial": "112/72 mmHg",
        "frequencia_cardiaca": "70 bpm",
        "frequencia_respiratoria": "15 irpm",
        "temperatura_c": "36.3",
        "saturacao_o2": "99%",
        "exame_fisico": "Mamas simetricas, sem nodulos palpaveis ou linfonodomegalia axilar. Exame ginecologico sem alteracoes.",
        "hipotese_diagnostica": "Rastreamento ginecologico/mamario de rotina — sem achados suspeitos.",
        "conduta": "Solicitadas mamografia de rastreamento e citologia oncotica (Papanicolau) de rotina; retorno anual.",
        "peso_kg": 62.0,
        "altura_cm": 166.0,
    },

# total pacientes novos: 30, ultimo exame id usado: 38
}

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
    4: {
        "data_solicitacao": "2026-09-03",
        "data_realizacao": "2026-09-03",
        "resultado": "Positivo para Influenza A, negativo para COVID-19.",
        "medico_solicitante": "Dra. Beatriz Ramos (Clinica Medica)",
        "observacoes": "Coletado na propria consulta, resultado em poucos minutos.",
    },
    5: {
        "data_solicitacao": "2026-09-02",
        "data_realizacao": "2026-09-02",
        "resultado": "Ritmo sinusal, sem sinais de sobrecarga ventricular ou isquemia aguda.",
        "medico_solicitante": "Dr. Bruno Castro (Cardiologia)",
        "observacoes": "Realizado na propria consulta.",
    },
    6: {
        "data_solicitacao": "2026-09-02",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Bruno Castro (Cardiologia)",
        "observacoes": "Reavaliacao de rotina do risco cardiovascular.",
    },
    7: {
        "data_solicitacao": "2026-09-03",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Juliana Prado (Endocrinologia)",
        "observacoes": "Reavaliacao de controle glicemico apos suspeita de descompensacao.",
    },
    8: {
        "data_solicitacao": "2026-09-03",
        "data_realizacao": "2026-09-04",
        "resultado": "256 mg/dL (valor de referencia: 70-99 mg/dL) — hiperglicemia significativa.",
        "medico_solicitante": "Dra. Juliana Prado (Endocrinologia)",
        "observacoes": "Coletado em jejum de 8 horas.",
    },
    9: {
        "data_solicitacao": "2026-09-03",
        "data_realizacao": "2026-09-03",
        "resultado": "Fratura completa da extremidade distal do radio direito, com desvio dorsal (compativel com fratura de Colles).",
        "medico_solicitante": "Dr. Eduardo Lima (Ortopedia)",
        "observacoes": "Realizado no mesmo dia do atendimento de urgencia.",
    },
    10: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Eduardo Lima (Ortopedia)",
        "observacoes": "Agendada apos reducao incruenta e imobilizacao definitiva.",
    },
    11: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": "2026-09-04",
        "resultado": "Leucocitose de 15.200/mm3 com desvio a esquerda, compativel com processo infeccioso/inflamatorio agudo.",
        "medico_solicitante": "Dr. Thiago Rocha (Cirurgia Geral)",
        "observacoes": "Coletado na admissao do pronto atendimento.",
    },
    12: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Thiago Rocha (Cirurgia Geral)",
        "observacoes": "Confirmacao de apendicite aguda antes da decisao cirurgica.",
    },
    13: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": "2026-09-04",
        "resultado": "Consolidacao em base pulmonar direita, compativel com pneumonia.",
        "medico_solicitante": "Dra. Fernanda Alves (Pneumologia)",
        "observacoes": "Realizado na admissao.",
    },
    14: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Fernanda Alves (Pneumologia)",
        "observacoes": "Avaliacao de gravidade e resposta inflamatoria.",
    },
    15: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Marcos Vieira (Urologia)",
        "observacoes": "Investigacao de infeccao urinaria recorrente.",
    },
    16: {
        "data_solicitacao": "2026-09-03",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Andre Barbosa (Neurologia)",
        "observacoes": "Exclusao de causas secundarias diante da piora do padrao de cefaleia.",
    },
    17: {
        "data_solicitacao": "2026-09-05",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Marcos Vieira (Urologia)",
        "observacoes": "Localizacao e dimensionamento do calculo para definir conduta.",
    },
    18: {
        "data_solicitacao": "2026-09-05",
        "data_realizacao": "2026-09-05",
        "resultado": "Hematuria microscopica presente, sem sinais de infeccao urinaria associada.",
        "medico_solicitante": "Dr. Marcos Vieira (Urologia)",
        "observacoes": "Coletado na admissao do pronto atendimento.",
    },
    19: {
        "data_solicitacao": "2026-09-05",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Beatriz Ramos (Clinica Medica)",
        "observacoes": "Solicitada como reserva, a depender da evolucao clinica.",
    },
    20: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Fernanda Alves (Pneumologia)",
        "observacoes": "Investigacao de tosse prolongada em paciente tabagista.",
    },
    21: {
        "data_solicitacao": "2026-09-05",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Patricia Souza (Infectologia)",
        "observacoes": "Confirmacao diagnostica apos contato domiciliar positivo.",
    },
    22: {
        "data_solicitacao": "2026-09-02",
        "data_realizacao": "2026-09-03",
        "resultado": "Hemoglobina 9.1 g/dL, VCM e HCM reduzidos — anemia microcitica e hipocromica.",
        "medico_solicitante": "Dra. Isabela Teixeira (Hematologia)",
        "observacoes": "Achados compativeis com anemia ferropriva.",
    },
    23: {
        "data_solicitacao": "2026-09-02",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Isabela Teixeira (Hematologia)",
        "observacoes": "Confirmacao dos estoques de ferro para fechar diagnostico.",
    },
    24: {
        "data_solicitacao": "2026-09-03",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Juliana Prado (Endocrinologia)",
        "observacoes": "Investigacao inicial de disfuncao tireoidiana.",
    },
    25: {
        "data_solicitacao": "2026-09-01",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Marina Cunha (Reumatologia)",
        "observacoes": "Investigacao de lombociatalgia persistente com sinais radiculares.",
    },
    26: {
        "data_solicitacao": "2026-09-05",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Patricia Souza (Infectologia)",
        "observacoes": "Confirmacao diagnostica; paciente orientado a retornar diariamente para reavaliacao.",
    },
    27: {
        "data_solicitacao": "2026-08-30",
        "data_realizacao": "2026-09-01",
        "resultado": "Nodulo espiculado em mama esquerda, BI-RADS 4C — achado suspeito, biopsia recomendada.",
        "medico_solicitante": "Dra. Renata Farias (Oncologia)",
        "observacoes": "Achado de alta suspeita, encaminhado para biopsia.",
    },
    28: {
        "data_solicitacao": "2026-09-02",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Renata Farias (Oncologia)",
        "observacoes": "Aguardando resultado anatomopatologico para definicao de conduta.",
    },
    29: {
        "data_solicitacao": "2026-08-31",
        "data_realizacao": "2026-09-01",
        "resultado": "Formacao cistica simples anecoica em mama direita, BI-RADS 2 (achado benigno).",
        "medico_solicitante": "Dr. Paulo Mendes (Radiologia)",
        "observacoes": "Sem necessidade de biopsia; seguimento de rotina.",
    },
    30: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": "2026-09-04",
        "resultado": "Neutrofilos e plaquetas dentro da faixa aceitavel para prosseguir com o ciclo.",
        "medico_solicitante": "Dra. Renata Farias (Oncologia)",
        "observacoes": "Controle de rotina antes de cada ciclo de quimioterapia.",
    },
    31: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Renata Farias (Oncologia)",
        "observacoes": "Avaliacao de resposta ao tratamento apos o 4º ciclo de quimioterapia.",
    },
    32: {
        "data_solicitacao": "2026-08-31",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Diego Almeida (Gastroenterologia)",
        "observacoes": "Avaliacao de esofagite e exclusao de esofago de Barrett dado o tempo de sintomas.",
    },
    33: {
        "data_solicitacao": "2026-09-01",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Thiago Rocha (Cirurgia Geral)",
        "observacoes": "Confirmacao do diagnostico e planejamento cirurgico eletivo.",
    },
    34: {
        "data_solicitacao": "2026-09-05",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dr. Gustavo Pires (Cirurgia Vascular)",
        "observacoes": "Prioridade alta — suspeita clinica de trombose venosa profunda.",
    },
    35: {
        "data_solicitacao": "2026-09-02",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Marina Cunha (Reumatologia)",
        "observacoes": "Exclusao de causas secundarias antes de fechar diagnostico de fibromialgia.",
    },
    36: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Beatriz Ramos (Clinica Medica)",
        "observacoes": "Check-up anual, sem queixas associadas.",
    },
    37: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Camila Nogueira (Ginecologia)",
        "observacoes": "Rastreamento anual de rotina.",
    },
    38: {
        "data_solicitacao": "2026-09-04",
        "data_realizacao": None,
        "resultado": None,
        "medico_solicitante": "Dra. Camila Nogueira (Ginecologia)",
        "observacoes": "Rastreamento anual de rotina.",
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


def _seed_anamnese(conn: sqlite3.Connection) -> None:
    """Insere a ficha de anamnese completa dos pacientes semente, uma por
    paciente_id (INSERT OR IGNORE respeita a UNIQUE em anamnese.paciente_id) —
    funciona tanto num banco novo (tabela recem-criada, vazia) quanto num banco
    existente que acabou de ganhar a tabela `anamnese` via CREATE TABLE IF NOT
    EXISTS. Nao sobrescreve uma anamnese ja registrada manualmente para o mesmo
    paciente.
    """
    colunas = [
        "paciente_id",
        "data_registro",
        "profissional_responsavel",
        "queixa_principal",
        "historia_doenca_atual",
        "historia_patologica_pregressa",
        "historia_familiar",
        "historia_ginecologica_obstetrica",
        "medicamentos_em_uso",
        "alergias",
        "cirurgias_previas",
        "tabagismo",
        "etilismo",
        "atividade_fisica",
        "pressao_arterial",
        "frequencia_cardiaca",
        "frequencia_respiratoria",
        "temperatura_c",
        "saturacao_o2",
        "peso_kg",
        "altura_cm",
        "exame_fisico",
        "hipotese_diagnostica",
        "conduta",
    ]
    placeholders = ", ".join("?" for _ in colunas)
    sql = f"INSERT OR IGNORE INTO anamnese ({', '.join(colunas)}) VALUES ({placeholders})"
    for paciente_id, ficha in _SEED_ANAMNESE.items():
        valores = [paciente_id] + [ficha[coluna] for coluna in colunas[1:]]
        conn.execute(sql, valores)


def init_mock_db(force: bool = False) -> None:
    """Cria o banco SQLite com schema e dados sinteticos, se ainda nao existir.
    Tambem migra e preenche os detalhes de exame em bancos ja existentes de uma
    versao anterior do schema (ver _migrar_schema_exames / _seed_exame_detalhes).

    force=True recria do zero (util em testes).

    Pacientes/exames semente sao inseridos com "INSERT OR IGNORE" (por id
    explicito, ver _SEED_PACIENTES/_SEED_EXAMES), nao mais so "se a tabela
    pacientes estiver vazia" — a versao antiga so semeava um banco totalmente
    vazio; quem ja tinha rodado o projeto antes (banco com os 2 pacientes
    originais) nunca ganhava pacientes/exames semente adicionados depois, porque
    "SELECT COUNT(*) FROM pacientes" ja retornava > 0. Com INSERT OR IGNORE por
    id, rodar de novo num banco existente so insere o que ainda nao existe (por
    id) e nao duplica nem sobrescreve o que ja esta la — mesmo padrao ja usado
    em _seed_anamnese (INSERT OR IGNORE) e _seed_paciente_detalhes/
    _seed_exame_detalhes (UPDATE ... COALESCE).
    """
    if force and DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executescript(_SCHEMA)
        _migrar_schema_pacientes(conn)
        _migrar_schema_exames(conn)

        conn.executemany(
            "INSERT OR IGNORE INTO pacientes (id, nome_ficticio, historico) VALUES (?, ?, ?)",
            _SEED_PACIENTES,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO exames (id, paciente_id, tipo, status) VALUES (?, ?, ?, ?)",
            _SEED_EXAMES,
        )

        _seed_paciente_detalhes(conn)
        _seed_exame_detalhes(conn)
        _seed_anamnese(conn)
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


_ANAMNESE_COLUNAS = [
    "id",
    "paciente_id",
    "data_registro",
    "profissional_responsavel",
    "queixa_principal",
    "historia_doenca_atual",
    "historia_patologica_pregressa",
    "historia_familiar",
    "historia_ginecologica_obstetrica",
    "medicamentos_em_uso",
    "alergias",
    "cirurgias_previas",
    "tabagismo",
    "etilismo",
    "atividade_fisica",
    "pressao_arterial",
    "frequencia_cardiaca",
    "frequencia_respiratoria",
    "temperatura_c",
    "saturacao_o2",
    "peso_kg",
    "altura_cm",
    "exame_fisico",
    "hipotese_diagnostica",
    "conduta",
]


def get_anamnese(paciente_id: int) -> dict | None:
    """Ficha de anamnese completa do paciente (queixa principal, historias,
    habitos, sinais vitais, exame fisico, hipotese diagnostica e conduta).
    Retorna None se o paciente ainda nao tiver anamnese registrada (ex.: paciente
    inserido manualmente sem passar pelo seed). Usado pelo endpoint
    GET /patients/{paciente_id}/anamnesis (ver api/main.py).
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            f"SELECT {', '.join(_ANAMNESE_COLUNAS)} FROM anamnese WHERE paciente_id = ?",
            (paciente_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(_ANAMNESE_COLUNAS, row))


# Rotulos legiveis (pt-BR) para cada campo da anamnese, na ordem em que devem
# aparecer no contexto enviado a LLM — mesma ordem clinica usada na tela do
# medico no front-end (queixa -> historias -> habitos -> sinais vitais ->
# exame fisico -> hipotese/conduta).
_ANAMNESE_ROTULOS = [
    ("queixa_principal", "Queixa principal"),
    ("historia_doenca_atual", "Historia da doenca atual"),
    ("historia_patologica_pregressa", "Historia patologica pregressa"),
    ("historia_familiar", "Historia familiar"),
    ("historia_ginecologica_obstetrica", "Historia ginecologica/obstetrica"),
    ("medicamentos_em_uso", "Medicamentos em uso"),
    ("alergias", "Alergias"),
    ("cirurgias_previas", "Cirurgias previas"),
    ("tabagismo", "Tabagismo"),
    ("etilismo", "Etilismo"),
    ("atividade_fisica", "Atividade fisica"),
    ("exame_fisico", "Exame fisico"),
    ("hipotese_diagnostica", "Hipotese diagnostica"),
    ("conduta", "Conduta"),
]

# Campos de sinais vitais, formatados numa unica linha condensada em vez de uma
# linha por campo (mais compacto e mais proximo de como um medico le sinais
# vitais num prontuario real).
# pressao_arterial/frequencia_cardiaca/frequencia_respiratoria/saturacao_o2 sao
# armazenados como texto ja com unidade (ex.: "76 bpm", "98%") — so peso/altura/
# temperatura sao numericos puros e precisam de sufixo aqui.
_SINAIS_VITAIS_CAMPOS = [
    ("pressao_arterial", "PA", ""),
    ("frequencia_cardiaca", "FC", ""),
    ("frequencia_respiratoria", "FR", ""),
    ("temperatura_c", "Temp", " C"),
    ("saturacao_o2", "SatO2", ""),
    ("peso_kg", "Peso", " kg"),
    ("altura_cm", "Altura", " cm"),
]


def get_demografia_paciente(paciente_id: int) -> dict | None:
    """Sexo, data de nascimento e idade calculada de um paciente.

    Existe separada de list_pacientes() porque o grafo precisa desses campos para
    UM paciente especifico em dois pontos distintos — montar o contexto do prompt
    e checar a coerencia etaria da resposta (agent/demographic_guard.py) — e
    filtrar a lista inteira para achar um id seria desperdicio.

    Retorna None se o paciente nao existir. `idade` e calculada a cada consulta a
    partir de data_nascimento (ver _calcular_idade), nunca armazenada.
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "SELECT id, nome_ficticio, sexo, data_nascimento FROM pacientes WHERE id = ?",
            (paciente_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "nome_ficticio": row[1],
        "sexo": row[2],
        "data_nascimento": row[3],
        "idade": _calcular_idade(row[3]),
    }


def montar_contexto_clinico(paciente_id: int) -> str | None:
    """Monta um unico bloco de texto com o historico resumido + a ficha de
    anamnese completa do paciente (quando existir), pronto para ser injetado no
    prompt da LLM como `patient_context` (ver rag/chain.py::ask).

    Usado pelo no `buscar_contexto_rag_e_gerar_resposta` do LangGraph
    (agent/nodes.py) para toda chamada a POST /assistant/ask — inclusive a
    "visao geral" automatica disparada pelo front-end ao abrir a tela do
    paciente: como o front so envia paciente_id + pergunta, e o backend quem
    busca a anamnese internamente aqui, sem exigir nada extra do chamador.

    Retorna None se o paciente nao tiver historico nem anamnese registrados
    (rag_ask trata None simplesmente omitindo essa secao do prompt).
    """
    partes: list[str] = []

    # Idade e sexo abrem o bloco de proposito. Sao os dois dados que mais
    # condicionam conduta clinica (a apresentacao, a investigacao e o
    # procedimento indicado para um mesmo diagnostico mudam completamente entre
    # uma crianca e um idoso), e ate entao nao chegavam ao modelo: nem
    # get_historico_paciente nem os campos de anamnese os contem. Sem esse dado
    # no prompt, um modelo pequeno preenche a lacuna com a faixa etaria mais
    # frequente na literatura do diagnostico — o que produz conduta correta para
    # o diagnostico e errada para o paciente.
    demografia = get_demografia_paciente(paciente_id)
    if demografia:
        identificacao = []
        if demografia.get("idade") is not None:
            identificacao.append(f"{demografia['idade']} anos")
        if demografia.get("sexo"):
            identificacao.append(f"sexo {demografia['sexo'].lower()}")
        if identificacao:
            partes.append("Paciente: " + ", ".join(identificacao) + ".")

    historico = get_historico_paciente(paciente_id)
    if historico:
        partes.append(f"Historico resumido: {historico}")

    anamnese = get_anamnese(paciente_id)
    if anamnese:
        for campo, rotulo in _ANAMNESE_ROTULOS:
            valor = anamnese.get(campo)
            if valor:
                partes.append(f"{rotulo}: {valor}")

        sinais = [
            f"{rotulo} {anamnese[campo]}{sufixo}"
            for campo, rotulo, sufixo in _SINAIS_VITAIS_CAMPOS
            if anamnese.get(campo) is not None
        ]
        if sinais:
            partes.append("Sinais vitais: " + ", ".join(sinais))

    if not partes:
        return None
    return "\n".join(partes)


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
