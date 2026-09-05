# Hospital VEinstein — Frontend

Frontend em **Angular 20** (standalone components + Signals + Angular Material) do projeto **OncoTech**, Tech Challenge Fase 3 da pós-graduação em IA (FIAP). É o painel usado pelo médico dentro do hospital para consultar pacientes, revisar a anamnese, acompanhar exames e conversar com o assistente médico virtual (backend em [`../1.AssistenteMedico`](../1.AssistenteMedico/README.md)).

> Plano de arquitetura completo do Tech Challenge em [`PLANO_Fase3.md`](../PLANO_Fase3.md), na raiz do repositório.

## Funcionalidades

- **Lista de pacientes** com busca por nome.
- **Ficha do paciente**: anamnese completa (queixa, históricos, hábitos, sinais vitais, exame físico, hipótese diagnóstica e conduta), organizada para leitura médica rápida.
- **Visão geral por IA**: ao abrir o paciente, o assistente já analisa a anamnese e os exames automaticamente, sugerindo condutas/exames e sinalizando pontos de preocupação com um alerta visual bem visível.
- **Chat com o assistente médico** (`/assistant/ask`) para perguntas livres sobre o paciente em consulta.
- **Exames**: realizados e pendentes (com detalhes do agendamento), lado a lado.

## Stack

- Angular 20 (standalone, sem NgModules) + Signals
- Angular Material 20 (tema `azure-blue`)
- RxJS

## Pré-requisitos

- Node.js 20+ e npm
- Backend do assistente rodando em `http://localhost:8001` (ver [`../1.AssistenteMedico/README.md`](../1.AssistenteMedico/README.md))

## Como rodar

```bash
cd front
npm install
npm start          # equivalente a `ng serve`
```

Acesse `http://localhost:4200`. A URL da API é configurada em `src/environments/environment.ts` (`apiUrl`).

## Build de produção

```bash
ng build
```

Os artefatos são gerados em `dist/front/`.

## Testes

```bash
ng test
```
