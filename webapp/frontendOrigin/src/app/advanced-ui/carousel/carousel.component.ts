import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-carousel',
  templateUrl: './carousel.component.html',
  styleUrls: ['./carousel.component.scss'],
})
export class CarouselComponent implements OnInit {

  constructor() { }

  ngOnInit() {
  }

  carouselOption1 = {
    loop: true,
    margin: 10,
    items: 1,
    nav: true,
    autoplay: true,
    autoplayTimeout: 5500,
    navText: ["<i class='mdi mdi-chevron-left'></i>", "<i class='mdi mdi-chevron-right'></i>"]
  }

  carouselOption2 = {
    center: true,
    items: 2,
    loop: true,
    margin: 10,
    autoplay: true,
    autoplayTimeout: 8500,
    responsive: {
      600: {
        items: 4
      }
    }
  }

  carouselOption3 = {
    items: 5,
    loop: false,
    margin: 10,
    autoplay: true,
    autoplayTimeout: 6000,
    responsive: {
      600: {
        items: 4
      }
    }
  }

  carouselOption4 = {
    items: 4,
    lazyLoad: true,
    loop: true,
    margin: 10,
    auto: true,
    autoplay: true,
    autoplayTimeout: 2500,
  }

  carouselOption5 = {
    rtl: true,
    loop: true,
    margin: 10,
    autoplay: true,
    autoplayTimeout: 3000,
    responsive: {
      0: {
        items: 1
      },
      600: {
        items: 3
      },
      1000: {
        items: 5
      }
    }
  }

}
