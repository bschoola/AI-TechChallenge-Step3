/** Espelha `AnamneseResponse` em `api/main.py` (GET /patients/{id}/anamnesis) —
 * ficha de anamnese completa registrada pela equipe médica. */
export interface Anamnesis {
  id: number;
  paciente_id: number;
  data_registro: string | null;
  profissional_responsavel: string | null;
  queixa_principal: string | null;
  historia_doenca_atual: string | null;
  historia_patologica_pregressa: string | null;
  historia_familiar: string | null;
  historia_ginecologica_obstetrica: string | null;
  medicamentos_em_uso: string | null;
  alergias: string | null;
  cirurgias_previas: string | null;
  tabagismo: string | null;
  etilismo: string | null;
  atividade_fisica: string | null;
  pressao_arterial: string | null;
  frequencia_cardiaca: string | null;
  frequencia_respiratoria: string | null;
  temperatura_c: string | null;
  saturacao_o2: string | null;
  peso_kg: number | null;
  altura_cm: number | null;
  exame_fisico: string | null;
  hipotese_diagnostica: string | null;
  conduta: string | null;
}
