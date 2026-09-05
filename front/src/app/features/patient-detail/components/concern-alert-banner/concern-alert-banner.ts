import { Component, computed, input } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';

/**
 * Alerta visual de destaque para pontos de preocupação sinalizados pela IA
 * (campo `alerta` e/ou `requer_validacao_humana` de AskResponse). Fica oculto
 * quando não há nada a sinalizar.
 */
@Component({
  selector: 'app-concern-alert-banner',
  imports: [MatIconModule],
  templateUrl: './concern-alert-banner.html',
  styleUrl: './concern-alert-banner.scss',
})
export class ConcernAlertBanner {
  readonly alerta = input<string | null>(null);
  readonly requerValidacaoHumana = input<boolean>(false);

  readonly visible = computed(() => !!this.alerta() || this.requerValidacaoHumana());
}
