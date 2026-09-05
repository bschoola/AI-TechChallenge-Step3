import { Component, inject, signal } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { debounceTime, distinctUntilChanged, startWith, switchMap } from 'rxjs';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTableModule } from '@angular/material/table';

import { PatientService } from '../../core/services/patient';
import { Patient } from '../../core/models/patient.model';

@Component({
  selector: 'app-patient-list',
  imports: [
    ReactiveFormsModule,
    MatButtonModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatProgressSpinnerModule,
    MatTableModule,
  ],
  templateUrl: './patient-list.html',
  styleUrl: './patient-list.scss',
})
export class PatientList {
  private readonly patientService = inject(PatientService);
  private readonly router = inject(Router);

  readonly searchControl = new FormControl('', { nonNullable: true });
  readonly patients = signal<Patient[]>([]);
  readonly loading = signal(true);
  readonly errorMessage = signal<string | null>(null);

  readonly displayedColumns = ['nome', 'idade', 'sexo', 'historico', 'acoes'];

  constructor() {
    this.searchControl.valueChanges
      .pipe(
        startWith(''),
        debounceTime(300),
        distinctUntilChanged(),
        switchMap((nome) => {
          this.loading.set(true);
          this.errorMessage.set(null);
          return this.patientService.list(nome || undefined);
        }),
        takeUntilDestroyed(),
      )
      .subscribe({
        next: (pacientes) => {
          this.patients.set(pacientes);
          this.loading.set(false);
        },
        error: () => {
          this.errorMessage.set(
            'Não foi possível carregar a lista de pacientes. Verifique se a API está em execução.',
          );
          this.loading.set(false);
        },
      });
  }

  openPatient(patient: Patient): void {
    this.router.navigate(['/pacientes', patient.id]);
  }
}
