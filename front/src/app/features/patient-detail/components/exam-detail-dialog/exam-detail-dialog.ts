import { Component, OnInit, inject, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

import { ExamService } from '../../../../core/services/exam';
import { ExamDetail } from '../../../../core/models/exam.model';

export interface ExamDetailDialogData {
  examId: number;
}

@Component({
  selector: 'app-exam-detail-dialog',
  imports: [MatButtonModule, MatDialogModule, MatIconModule, MatProgressSpinnerModule],
  templateUrl: './exam-detail-dialog.html',
  styleUrl: './exam-detail-dialog.scss',
})
export class ExamDetailDialog implements OnInit {
  private readonly examService = inject(ExamService);
  private readonly data = inject<ExamDetailDialogData>(MAT_DIALOG_DATA);
  private readonly dialogRef = inject(MatDialogRef<ExamDetailDialog>);

  readonly loading = signal(true);
  readonly exam = signal<ExamDetail | null>(null);
  readonly errorMessage = signal<string | null>(null);

  ngOnInit(): void {
    this.examService.getById(this.data.examId).subscribe({
      next: (exam) => {
        this.exam.set(exam);
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set('Não foi possível carregar o detalhe deste exame.');
        this.loading.set(false);
      },
    });
  }

  close(): void {
    this.dialogRef.close();
  }
}
