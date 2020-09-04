import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-advanced-elements',
  templateUrl: './advanced-elements.component.html',
  styleUrls: ['./advanced-elements.component.scss']
})
export class AdvancedElementsComponent implements OnInit {

  fontAwesomeRate = "2";
  cssRate = 1;
  oneToTenRate = 6;
  squareRate = 3;
  verticalRate = 3;
  bootstrapRate = 3;
  movieRate = 3;
  pickerColor = "#b66dff";

  date: Date = new Date();
  datePickersettings = {
    bigBanner: true,
    timePicker: true,
    format: 'dd-MM-yyyy',
    defaultOpen: false
  }
  tagInputItems = ['Pizza', 'Pasta', 'Parmesan'];

  constructor() { }

  ngOnInit() {
  }

}
