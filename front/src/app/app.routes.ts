import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    loadComponent: () =>
      import('./features/patient-list/patient-list').then((m) => m.PatientList),
  },
  {
    path: 'pacientes/:id',
    loadComponent: () =>
      import('./features/patient-detail/patient-detail').then((m) => m.PatientDetail),
  },
  { path: '**', redirectTo: '' },
];
