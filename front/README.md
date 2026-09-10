# Hospital VEinstein: Frontend

Frontend em **Angular 20**, com standalone components, Signals e Angular Material, da entrega da Fase 3 do Tech Challenge da pós-graduação em IA (FIAP). É o painel usado pelo médico dentro do hospital para consultar pacientes, revisar a anamnese, acompanhar exames e conversar com o assistente médico virtual. O backend está em [`../1.AssistenteMedico`](../1.AssistenteMedico/README.md).

> O plano de arquitetura completo do Tech Challenge está em [`PLANO_Fase3.md`](../PLANO_Fase3.md), na raiz do repositório.

## Funcionalidades

- **Lista de pacientes** com busca por nome.
- **Ficha do paciente**, com a anamnese completa (queixa, históricos, hábitos, sinais vitais, exame físico, hipótese diagnóstica e conduta), organizada para leitura médica rápida.
- **Visão geral por IA**. Ao abrir o paciente, o assistente já analisa a anamnese e os exames automaticamente, sugerindo condutas e exames e sinalizando pontos de preocupação com um alerta visual bem visível.
- **Chat com o assistente médico** (`/assistant/ask`), para perguntas livres sobre o paciente em consulta.
- **Exames** realizados e pendentes, com os detalhes do agendamento, lado a lado.

## Stack

- Angular 20 (standalone, sem NgModules) com Signals
- Angular Material 20 (tema `azure-blue`)
- RxJS

## Pré-requisitos

- Node.js 20 ou superior, e npm
- Backend do assistente rodando em `http://localhost:8001`. Veja [`../1.AssistenteMedico/README.md`](../1.AssistenteMedico/README.md)

## Como rodar

```bash
cd front
npm install
npm start          # equivalente a `ng serve`
```

Acesse `http://localhost:4200`. A URL da API é configurada em `src/environments/environment.ts`, na chave `apiUrl`.

## Build de produção

```bash
ng build
```

Os artefatos são gerados em `dist/front/`.

## Testes

```bash
ng test
```
