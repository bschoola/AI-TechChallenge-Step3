import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { environment } from '../../../environments/environment';
import { Patient } from '../models/patient.model';

/**
 * A API não expõe um GET /patients/{id} dedicado — apenas GET /patients (lista
 * completa, com filtro opcional por nome). Por isso `getById` busca a lista e
 * filtra no cliente; para o volume de pacientes do prontuário mock isso é
 * suficiente e evita duplicar lógica de busca no backend.
 */
@Injectable({
  providedIn: 'root',
})
export class PatientService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/patients`;

  list(nome?: string): Observable<Patient[]> {
    const params: Record<string, string> = nome ? { nome } : {};
    return this.http.get<Patient[]>(this.baseUrl, { params });
  }

  getById(id: number): Observable<Patient | undefined> {
    return this.list().pipe(map((pacientes) => pacientes.find((p) => p.id === id)));
  }
}
