import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import { ExamDetail, ExamSummary } from '../models/exam.model';

@Injectable({
  providedIn: 'root',
})
export class ExamService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiUrl;

  listByPatient(patientId: number): Observable<ExamSummary[]> {
    return this.http.get<ExamSummary[]>(`${this.baseUrl}/patients/${patientId}/exams`);
  }

  getById(examId: number): Observable<ExamDetail> {
    return this.http.get<ExamDetail>(`${this.baseUrl}/exams/${examId}`);
  }
}
