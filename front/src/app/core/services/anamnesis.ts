import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import { Anamnesis } from '../models/anamnesis.model';

@Injectable({
  providedIn: 'root',
})
export class AnamnesisService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiUrl;

  /** GET /patients/{id}/anamnesis — 404 tanto se o paciente não existir quanto
   * se existir mas ainda não tiver anamnese registrada (o chamador distingue
   * pelo `detail`, se precisar). */
  getByPatientId(patientId: number): Observable<Anamnesis> {
    return this.http.get<Anamnesis>(`${this.baseUrl}/patients/${patientId}/anamnesis`);
  }
}
