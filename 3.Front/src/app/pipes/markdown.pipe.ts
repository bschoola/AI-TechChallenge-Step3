import { Pipe, PipeTransform } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

/**
 * Converte markdown básico em HTML seguro para uso com [innerHTML].
 * Suporta: **negrito**, *itálico* e quebras de linha.
 */
@Pipe({ name: 'markdown', standalone: true })
export class MarkdownPipe implements PipeTransform {
  constructor(private sanitizer: DomSanitizer) {}

  transform(value: string | null | undefined): SafeHtml {
    if (!value) return '';

    const html = value
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')   // **negrito**
      .replace(/\*(.+?)\*/g, '<em>$1</em>')                // *itálico*
      .replace(/\n/g, '<br>');                             // quebras de linha

    return this.sanitizer.bypassSecurityTrustHtml(html);
  }
}
