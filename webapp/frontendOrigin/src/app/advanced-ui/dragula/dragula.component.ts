import { Component, OnInit } from '@angular/core';
import { DragulaService } from 'ng2-dragula';

@Component({
  selector: 'app-dragula',
  templateUrl: './dragula.component.html',
  styleUrls: ['./dragula.component.scss'],
  providers: [DragulaService]
})
export class DragulaComponent implements OnInit {

  constructor() { }

  ngOnInit() {
  }

}
