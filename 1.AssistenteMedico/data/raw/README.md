# Dados brutos (sintéticos)

Coloque aqui os documentos-fonte fictícios, um arquivo `.md` ou `.txt` por documento, organizados em subpastas por tipo:

```
raw/
├── protocolos/     # protocolos clínicos por especialidade/condição (30-50 documentos)
├── faqs/           # perguntas frequentes de médicos + resposta padrão
└── laudos_modelo/  # modelos de laudo e receita (formato/tom, para o fine-tuning)
```

Todos os dados aqui são **sintéticos** (gerados com apoio de IA generativa e curados manualmente) — não há paciente, médico ou hospital real. Isso substitui a etapa de "anonimização" pedida no desafio: não há PII para anonimizar porque nada é real por construção. Documentar essa decisão no relatório técnico.

Ver `PLANO_Fase3.md` (raiz do repo), seção 3, para o formato sugerido de cada tipo de documento.
