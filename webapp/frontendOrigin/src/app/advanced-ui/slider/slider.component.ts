import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-slider',
  templateUrl: './slider.component.html',
  styleUrls: ['./slider.component.scss']
})
export class SliderComponent implements OnInit {

  constructor() { }

  ngOnInit() {
  }

  range1 = 72;
  range2 = 92;
  range3 = 43;
  range4 = 20;
  range5 = 75;
  range6 = 72;
  range7 = 92;
  range8 = 43;
  range9 = 20;
  range10 = 75;
  range11 = 75;
  range12 = [20, 80];
  range13 = [20, 80];
  range14 = [20, 80];
  range15 = [23.4, 54.3];
  range16 = [29.1, 74.1];
  range17 = [13.4, 89.3];

  verticalSliderConfig = {
    orientation: 'vertical'
  };

  tooltipSliderConfig = {
    connect: true,
    tooltips: true,
    range: {
      min: 0,
      max: 100
    },
    pips: {
      mode: 'range',
      density: 5
    }
  };

}
