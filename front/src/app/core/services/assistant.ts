import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import { AskRequest, AskResponse, OverviewResponse } from '../models/assistant.model';

@Injectable({
  providedIn: 'root',
})
export class AssistantService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiUrl;

  /** Pergunta livre do médico ao assistente (chat) — POST /assistant/ask. */
  ask(request: AskRequest): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${this.baseUrl}/assistant/ask`, request);
  }

  /** Visão geral automática do paciente (card de IA da tela de detalhe) —
   * POST /patients/{id}/overview. Endpoint dedicado (em vez de reusar /assistant/ask
   * com um prompt fixo de "visão geral"): o modelo, ao ser instruído a "revisar a
   * anamnese" dentro do mesmo fluxo do chat, tendia só a parafrasear a ficha que já
   * aparece do lado esquerdo da tela em vez de destacar o que é relevante. Esse
   * endpoint usa um nó/prompt dedicados no backend (agent/nodes.py::gerar_visao_geral)
   * que devolve pontos_relevantes/pontos_atencao já estruturados. */
  getOverview(patientId: number): Observable<OverviewResponse> {
    return this.http.post<OverviewResponse>(`${this.baseUrl}/patients/${patientId}/overview`, {});
  }
}
