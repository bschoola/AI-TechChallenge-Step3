import { Component, computed, input, output } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

import { ExamSummary } from '../../../../core/models/exam.model';

@Component({
  selector: 'app-exams-panel',
  imports: [MatButtonModule, MatCardModule, MatIconModule, MatProgressSpinnerModule],
  templateUrl: './exams-panel.html',
  styleUrl: './exams-panel.scss',
})
export class ExamsPanel {
  readonly exams = input<ExamSummary[]>([]);
  readonly loading = input(false);
  readonly errorMessage = input<string | null>(null);

  readonly examSelected = output<number>();

  readonly concluidos = computed(() => this.exams().filter((e) => e.status === 'concluido'));
  readonly pendentes = computed(() => this.exams().filter((e) => e.status === 'pendente'));

  select(exam: ExamSummary): void {
    this.examSelected.emit(exam.id);
  }
}
