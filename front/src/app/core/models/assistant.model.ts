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

/** Uma entrada do histórico local de chat da tela do paciente (não persiste no backend —
 * cada chamada a /assistant/ask é independente/sem memória de conversa). */
export interface ChatEntry {
  pergunta: string;
  resposta?: AskResponse;
  loading: boolean;
  erro?: string;
  timestamp: Date;
}
