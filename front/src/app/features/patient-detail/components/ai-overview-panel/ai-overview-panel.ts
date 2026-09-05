import {
  Component,
  effect,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';

import { AssistantService } from '../../../../core/services/assistant';
import { OverviewResponse } from '../../../../core/models/assistant.model';

/**
 * Ao receber um patientId, chama automaticamente POST /patients/{id}/overview,
 * que devolve uma síntese já estruturada em pontos relevantes e pontos de atenção
 * (em vez de um parágrafo livre) — ver AssistantService.getOverview e
 * agent/nodes.py::gerar_visao_geral no backend. Emite `overview` para o pai
 * renderizar o alerta visual no topo da tela quando aplicável.
 */
@Component({
  selector: 'app-ai-overview-panel',
  imports: [
    MatButtonModule,
    MatCardModule,
    MatChipsModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
  ],
  templateUrl: './ai-overview-panel.html',
  styleUrl: './ai-overview-panel.scss',
})
export class AiOverviewPanel {
  private readonly assistantService = inject(AssistantService);

  readonly patientId = input.required<number>();
  readonly overview = output<OverviewResponse | null>();

  readonly loading = signal(true);
  readonly response = signal<OverviewResponse | null>(null);
  readonly errorMessage = signal<string | null>(null);

  constructor() {
    effect((onCleanup) => {
      const id = this.patientId();
      let cancelled = false;
      onCleanup(() => {
        cancelled = true;
      });
      this.load(id, () => cancelled);
    });
  }

  reload(): void {
    const id = this.patientId();
    this.load(id, () => false);
  }

  private load(patientId: number, isCancelled: () => boolean): void {
    this.loading.set(true);
    this.errorMessage.set(null);
    this.response.set(null);
    this.overview.emit(null);

    this.assistantService.getOverview(patientId).subscribe({
      next: (res) => {
        if (isCancelled()) return;
        this.response.set(res);
        this.loading.set(false);
        this.overview.emit(res);
      },
      error: (err) => {
        if (isCancelled()) return;
        this.errorMessage.set(this.describeError(err));
        this.loading.set(false);
        this.overview.emit(null);
      },
    });
  }

  private describeError(err: unknown): string {
    const status = (err as { status?: number })?.status;
    const detail = (err as { error?: { detail?: string } })?.error?.detail;
    if (status === 503) {
      return (
        detail ??
        'O assistente de IA ainda não está pronto neste ambiente (faltam artefatos de RAG/fine-tuning).'
      );
    }
    return detail ?? 'Não foi possível gerar a visão geral de IA para este paciente agora.';
  }
}
