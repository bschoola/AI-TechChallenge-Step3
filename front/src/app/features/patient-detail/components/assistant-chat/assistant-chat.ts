import { Component, ElementRef, inject, input, signal, ViewChild } from '@angular/core';
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

  @ViewChild('questionInput') private questionInput?: ElementRef<HTMLTextAreaElement>;

  /** Altura máxima (px) até onde o campo cresce sozinho ao digitar; acima
   * disso o usuário ainda pode arrastar a alça de redimensionamento manualmente. */
  private readonly maxAutoGrowHeight = 160;

  send(): void {
    const pergunta = this.question.trim();
    if (!pergunta) return;

    const entry: ChatEntry = { pergunta, loading: true, timestamp: new Date() };
    this.entries.update((list) => [...list, entry]);
    this.question = '';
    this.resetInputHeight();

    this.assistantService.ask({ paciente_id: this.patientId(), pergunta }).subscribe({
      next: (resposta) => {
        this.updateEntry(entry, { resposta, loading: false });
      },
      error: (err) => {
        this.updateEntry(entry, { loading: false, erro: this.describeError(err) });
      },
    });
  }

  /** Envia com Enter; permite quebra de linha com Shift+Enter. */
  onQuestionKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  /** Cresce o textarea conforme o conteúdo digitado, até maxAutoGrowHeight. */
  autoGrowInput(textarea: HTMLTextAreaElement): void {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, this.maxAutoGrowHeight)}px`;
  }

  private resetInputHeight(): void {
    const el = this.questionInput?.nativeElement;
    if (el) {
      el.style.height = 'auto';
    }
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
