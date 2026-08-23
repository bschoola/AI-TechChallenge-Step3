import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, interval, EMPTY } from 'rxjs';
import { switchMap, takeWhile, catchError } from 'rxjs/operators';

export interface PredictionInput {
  area_pior: number;
  textura_pior: number;
  pontos_concavos_pior: number;
  concavidade_pior: number;
}

export interface PredictionResponse {
  diagnostico: string;
  confianca: number;
  classe: number;
  probabilidade_maligno: number;
  probabilidade_benigno: number;
}

/** Resposta do POST /predict/breastCancer/laudo — retorna task_id imediatamente */
export interface LaudoAsyncResponse extends PredictionResponse {
  task_id: string;
}

/** Resposta do GET /predict/breastCancer/laudo/status/{task_id} */
export interface LaudoStatusResponse {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  laudo?: string;
  error?: string;
}

const POLL_INTERVAL_MS = 2000;

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  predict(input: PredictionInput): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(
      `${this.apiUrl}/predict/breastCancer`,
      input
    );
  }

  /**
   * Dispara a geração do laudo e retorna a predição + task_id imediatamente.
   * Use pollLaudo(task_id) para acompanhar o progresso.
   */
  requestLaudo(input: PredictionInput): Observable<LaudoAsyncResponse> {
    return this.http.post<LaudoAsyncResponse>(
      `${this.apiUrl}/predict/breastCancer/laudo`,
      input
    );
  }

  /**
   * Faz polling a cada 2s até o laudo ficar pronto (completed) ou falhar.
   * Emite cada resposta de status e completa ao atingir estado terminal.
   */
  pollLaudo(taskId: string): Observable<LaudoStatusResponse> {
    return interval(POLL_INTERVAL_MS).pipe(
      switchMap(() =>
        this.http.get<LaudoStatusResponse>(
          `${this.apiUrl}/predict/breastCancer/laudo/status/${taskId}`
        ).pipe(catchError(() => EMPTY))
      ),
      takeWhile(
        (r) => r.status !== 'completed' && r.status !== 'failed',
        true // inclui o último emit (o que encerrou o takeWhile)
      )
    );
  }
}
