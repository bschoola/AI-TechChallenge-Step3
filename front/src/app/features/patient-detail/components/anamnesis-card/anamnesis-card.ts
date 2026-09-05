import { Component, computed, effect, inject, input, signal } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

import { AnamnesisService } from '../../../../core/services/anamnesis';
import { Anamnesis } from '../../../../core/models/anamnesis.model';

interface VitalSign {
  label: string;
  value: string;
}

/**
 * Card principal da tela do paciente — ficha de anamnese completa vinda de
 * GET /patients/{id}/anamnesis, organizada em seções para leitura médica
 * rápida (queixa, históricos, hábitos/medicações, sinais vitais, exame
 * físico, hipótese diagnóstica e conduta). Maior peso visual da tela,
 * conforme prioridade definida com o usuário: anamnese > visão da IA > chat.
 */
@Component({
  selector: 'app-anamnesis-card',
  imports: [MatIconModule, MatProgressSpinnerModule],
  templateUrl: './anamnesis-card.html',
  styleUrl: './anamnesis-card.scss',
})
export class AnamnesisCard {
  private readonly anamnesisService = inject(AnamnesisService);

  readonly patientId = input.required<number>();

  readonly loading = signal(true);
  readonly anamnesis = signal<Anamnesis | null>(null);
  readonly notRegistered = signal(false);
  readonly errorMessage = signal<string | null>(null);

  readonly vitalSigns = computed<VitalSign[]>(() => {
    const a = this.anamnesis();
    if (!a) return [];
    const sinais: VitalSign[] = [];
    if (a.pressao_arterial) sinais.push({ label: 'PA', value: a.pressao_arterial });
    if (a.frequencia_cardiaca) sinais.push({ label: 'FC', value: a.frequencia_cardiaca });
    if (a.frequencia_respiratoria) sinais.push({ label: 'FR', value: a.frequencia_respiratoria });
    if (a.temperatura_c) sinais.push({ label: 'Temp.', value: `${a.temperatura_c} °C` });
    if (a.saturacao_o2) sinais.push({ label: 'SatO₂', value: a.saturacao_o2 });
    if (a.peso_kg !== null) sinais.push({ label: 'Peso', value: `${a.peso_kg} kg` });
    if (a.altura_cm !== null) sinais.push({ label: 'Altura', value: `${a.altura_cm} cm` });
    return sinais;
  });

  constructor() {
    effect((onCleanup) => {
      const id = this.patientId();
      let cancelled = false;
      onCleanup(() => {
        cancelled = true;
      });

      this.loading.set(true);
      this.notRegistered.set(false);
      this.errorMessage.set(null);
      this.anamnesis.set(null);

      this.anamnesisService.getByPatientId(id).subscribe({
        next: (anamnesis) => {
          if (cancelled) return;
          this.anamnesis.set(anamnesis);
          this.loading.set(false);
        },
        error: (err) => {
          if (cancelled) return;
          if (err?.status === 404) {
            this.notRegistered.set(true);
          } else {
            this.errorMessage.set('Não foi possível carregar a anamnese deste paciente.');
          }
          this.loading.set(false);
        },
      });
    });
  }
}
