/** Espelha `ExameResumoResponse` em `api/main.py` (GET /patients/{id}/exams). */
export interface ExamSummary {
  id: number;
  tipo: string;
  status: 'pendente' | 'concluido';
  data_solicitacao: string | null;
  /** Permanece null enquanto o exame estiver pendente. */
  data_realizacao: string | null;
}

/** Espelha `ExameDetailResponse` em `api/main.py` (GET /exams/{id}). */
export interface ExamDetail extends ExamSummary {
  paciente_id: number;
  resultado: string | null;
  medico_solicitante: string | null;
  observacoes: string | null;
}
