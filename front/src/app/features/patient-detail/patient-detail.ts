import { Component, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { switchMap } from 'rxjs';

import { MatButtonModule } from '@angular/material/button';
import { MatDialog } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

import { PatientService } from '../../core/services/patient';
import { ExamService } from '../../core/services/exam';
import { Patient } from '../../core/models/patient.model';
import { ExamSummary } from '../../core/models/exam.model';
import { OverviewResponse } from '../../core/models/assistant.model';

import { AnamnesisCard } from './components/anamnesis-card/anamnesis-card';
import { ConcernAlertBanner } from './components/concern-alert-banner/concern-alert-banner';
import { AiOverviewPanel } from './components/ai-overview-panel/ai-overview-panel';
import { AssistantChat } from './components/assistant-chat/assistant-chat';
import { ExamsPanel } from './components/exams-panel/exams-panel';
import {
  ExamDetailDialog,
  ExamDetailDialogData,
} from './components/exam-detail-dialog/exam-detail-dialog';

@Component({
  selector: 'app-patient-detail',
  imports: [
    RouterLink,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    AnamnesisCard,
    ConcernAlertBanner,
    AiOverviewPanel,
    AssistantChat,
    ExamsPanel,
  ],
  templateUrl: './patient-detail.html',
  styleUrl: './patient-detail.scss',
})
export class PatientDetail {
  private readonly route = inject(ActivatedRoute);
  private readonly patientService = inject(PatientService);
  private readonly examService = inject(ExamService);
  private readonly dialog = inject(MatDialog);

  readonly patientId = signal<number | null>(null);
  readonly patient = signal<Patient | undefined>(undefined);
  readonly patientLoading = signal(true);
  readonly patientNotFound = signal(false);

  readonly exams = signal<ExamSummary[]>([]);
  readonly examsLoading = signal(true);
  readonly examsError = signal<string | null>(null);

  readonly overviewResponse = signal<OverviewResponse | null>(null);

  readonly pendentesCount = computed(
    () => this.exams().filter((e) => e.status === 'pendente').length,
  );
  readonly concluidosCount = computed(
    () => this.exams().filter((e) => e.status === 'concluido').length,
  );

  constructor() {
    this.route.paramMap
      .pipe(
        switchMap((params) => {
          const id = Number(params.get('id'));
          this.patientId.set(id);
          this.patientLoading.set(true);
          this.patientNotFound.set(false);
          this.examsLoading.set(true);
          this.examsError.set(null);
          this.loadExams(id);
          return this.patientService.getById(id);
        }),
        takeUntilDestroyed(),
      )
      .subscribe({
        next: (patient) => {
          this.patient.set(patient);
          this.patientNotFound.set(!patient);
          this.patientLoading.set(false);
        },
        error: () => {
          this.patientNotFound.set(true);
          this.patientLoading.set(false);
        },
      });
  }

  onOverview(response: OverviewResponse | null): void {
    this.overviewResponse.set(response);
  }

  openExam(examId: number): void {
    this.dialog.open<ExamDetailDialog, ExamDetailDialogData>(ExamDetailDialog, {
      data: { examId },
      width: '520px',
      autoFocus: false,
    });
  }

  private loadExams(patientId: number): void {
    this.examService.listByPatient(patientId).subscribe({
      next: (exams) => {
        this.exams.set(exams);
        this.examsLoading.set(false);
      },
      error: () => {
        this.examsError.set('Não foi possível carregar os exames deste paciente.');
        this.examsLoading.set(false);
      },
    });
  }
}
