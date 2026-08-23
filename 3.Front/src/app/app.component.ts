import { Component, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import {
  PredictionService,
  PredictionInput,
  PredictionResponse,
} from './services/prediction.service';
import { MarkdownPipe } from './pipes/markdown.pipe';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, MarkdownPipe],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnDestroy {
  form: PredictionInput = {
    area_pior: null!,
    textura_pior: null!,
    pontos_concavos_pior: null!,
    concavidade_pior: null!,
  };

  result: PredictionResponse | null = null;
  laudo: string | null = null;
  loadingPredict = false;
  loadingLaudo   = false;
  error: string | null = null;
  errorLaudo: string | null = null;
  dataAtual = new Date();

  private _pollSub: Subscription | null = null;

  constructor(private predictionService: PredictionService) {}

  ngOnDestroy(): void {
    this._pollSub?.unsubscribe();
  }

  submit(): void {
    this._pollSub?.unsubscribe();
    this.loadingPredict = true;
    this.loadingLaudo   = false;
    this.error          = null;
    this.errorLaudo     = null;
    this.result         = null;
    this.laudo          = null;
    this.dataAtual      = new Date();

    // Passo 1: classificador rápido
    this.predictionService.predict(this.form).subscribe({
      next: (res) => {
        this.result         = res;
        this.loadingPredict = false;

        // Passo 2: dispara geração assíncrona do laudo
        this.loadingLaudo = true;
        this.predictionService.requestLaudo(this.form).subscribe({
          next: (laudoAsync) => {
            // Passo 3: polling até laudo pronto
            this._pollSub = this.predictionService
              .pollLaudo(laudoAsync.task_id)
              .subscribe({
                next: (status) => {
                  if (status.status === 'completed') {
                    this.laudo        = status.laudo ?? null;
                    this.loadingLaudo = false;
                  } else if (status.status === 'failed') {
                    this.errorLaudo   = status.error ?? 'Erro ao gerar o laudo.';
                    this.loadingLaudo = false;
                  }
                },
                error: () => {
                  this.errorLaudo   = 'Erro ao verificar status do laudo.';
                  this.loadingLaudo = false;
                },
              });
          },
          error: () => {
            this.errorLaudo   = 'Não foi possível iniciar a geração do laudo.';
            this.loadingLaudo = false;
          },
        });
      },
      error: () => {
        this.error =
          'Não foi possível conectar à API. Verifique se o servidor está em execução em http://localhost:8000.';
        this.loadingPredict = false;
      },
    });
  }

  novaAnalise(): void {
    this._pollSub?.unsubscribe();
    this.result         = null;
    this.laudo          = null;
    this.error          = null;
    this.errorLaudo     = null;
    this.loadingPredict = false;
    this.loadingLaudo   = false;
    this.form = {
      area_pior: null!,
      textura_pior: null!,
      pontos_concavos_pior: null!,
      concavidade_pior: null!,
    };
  }

  printLaudo(): void {
    window.print();
  }

  get isMaligno(): boolean {
    return this.result?.diagnostico === 'Maligno';
  }

  get barraMalignoWidth(): string {
    return `${(this.result?.probabilidade_maligno ?? 0) * 100}%`;
  }

  get barraBenignoWidth(): string {
    return `${(this.result?.probabilidade_benigno ?? 0) * 100}%`;
  }
}
