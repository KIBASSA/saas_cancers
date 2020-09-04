import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-clipboard',
  templateUrl: './clipboard.component.html',
  styleUrls: ['./clipboard.component.scss']
})
export class ClipboardComponent implements OnInit {

  constructor() { }

  ngOnInit() {
  }

  // Clear input text after copy
  cut(input) {
    input.value = '';
  }

}
