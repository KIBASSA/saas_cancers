import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-wizard',
  templateUrl: './wizard.component.html',
  styleUrls: ['./wizard.component.scss']
})
export class WizardComponent implements OnInit {
  isCompleted: boolean;

  constructor() { }

  ngOnInit() {
  }

  onComplete(event) {
    this.isCompleted = true;
  }

}
