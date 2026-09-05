import { Component, inject, input, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

import { AssistantService } from '../../../../core/services/assistant';
import { ChatEntry } from '../../../../core/models/assistant.model';

/**
 * Chat livre do médico com o assistente para a consulta em andamento. Cada
 * pergunta é uma chamada independente a POST /assistant/ask (a API não mantém
 * memória de conversa) — o histórico exibido é só local, desta sessão de tela.
 * Tratado com peso visual secundário na tela (menor prioridade que a anamnese
 * e a visão geral da IA).
 */
@Component({
  selector: 'app-assistant-chat',
  imports: [FormsModule, MatCardModule, MatIconModule, MatProgressSpinnerModule],
  templateUrl: './assistant-chat.html',
  styleUrl: './assistant-chat.scss',
})
export class AssistantChat {
  private readonly assistantService = inject(AssistantService);

  readonly patientId = input.required<number>();

  readonly entries = signal<ChatEntry[]>([]);
  question = '';

  send(): void {
    const pergunta = this.question.trim();
    if (!pergunta) return;

    const entry: ChatEntry = { pergunta, loading: true, timestamp: new Date() };
    this.entries.update((list) => [...list, entry]);
    this.question = '';

    this.assistantService.ask({ paciente_id: this.patientId(), pergunta }).subscribe({
      next: (resposta) => {
        this.updateEntry(entry, { resposta, loading: false });
      },
      error: (err) => {
        this.updateEntry(entry, { loading: false, erro: this.describeError(err) });
      },
    });
  }

  private updateEntry(target: ChatEntry, changes: Partial<ChatEntry>): void {
    this.entries.update((list) =>
      list.map((item) => (item === target ? { ...item, ...changes } : item)),
    );
  }

  private describeError(err: unknown): string {
    const detail = (err as { error?: { detail?: string } })?.error?.detail;
    return detail ?? 'Não foi possível obter uma resposta do assistente agora.';
  }
}
