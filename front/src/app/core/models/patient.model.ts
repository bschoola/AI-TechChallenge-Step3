/** Espelha `PacienteResponse` em `api/main.py` (GET /patients). */
export interface Patient {
  id: number;
  nome_ficticio: string;
  historico: string | null;
  sexo: string | null;
  data_nascimento: string | null;
  /** Calculada no backend a partir de data_nascimento — não é armazenada. */
  idade: number | null;
}
