import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { TopBar } from './shared/components/top-bar/top-bar';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, TopBar],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {}
