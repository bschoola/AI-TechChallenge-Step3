import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import { AskRequest, AskResponse } from '../models/assistant.model';

/** Prompt padrão disparado automaticamente ao abrir a tela do paciente, para
 * trazer uma visão geral gerada pela IA (anamnese, exames recentes, sugestões
 * e pontos de preocupação) antes de qualquer pergunta manual do médico. */
export const VISAO_GERAL_PROMPT =
  'Faça uma visão geral deste paciente para a equipe médica: revise a anamnese, ' +
  'analise os exames realizados recentemente, sugira exames adicionais se ' +
  'pertinente considerando o histórico e os exames pendentes, e sinalize ' +
  'claramente quaisquer pontos de preocupação que mereçam atenção.';

@Injectable({
  providedIn: 'root',
})
export class AssistantService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/assistant`;

  ask(request: AskRequest): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${this.baseUrl}/ask`, request);
  }
}
