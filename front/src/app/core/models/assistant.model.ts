/** Espelha `AskRequest` em `api/main.py` (POST /assistant/ask). */
export interface AskRequest {
  paciente_id: number;
  pergunta: string;
}

/** Espelha `AskResponse` em `api/main.py` (POST /assistant/ask). */
export interface AskResponse {
  resposta: string;
  fontes: string[];
  alerta: string | null;
  requer_validacao_humana: boolean;
  status: string;
}

/** Espelha `VisaoGeralResponse` em `api/main.py` (POST /patients/{id}/overview).
 * Diferente do chat, a visão geral vem estruturada em duas listas de tópicos —
 * pontos relevantes e pontos de atenção — em vez de um parágrafo único, para o
 * card não repetir a ficha de anamnese que já aparece do lado esquerdo da tela. */
export interface OverviewResponse {
  pontos_relevantes: string[];
  pontos_atencao: string[];
  fontes: string[];
  alerta: string | null;
  requer_validacao_humana: boolean;
  status: string;
}

/** Uma entrada do histórico local de chat da tela do paciente (não persiste no backend —
 * cada chamada a /assistant/ask é independente/sem memória de conversa). */
export interface ChatEntry {
  pergunta: string;
  resposta?: AskResponse;
  loading: boolean;
  erro?: string;
  timestamp: Date;
}
