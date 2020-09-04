import { Component, OnInit, ViewChild } from '@angular/core';
import { SwalComponent } from '@sweetalert2/ngx-sweetalert2';

@Component({
  selector: 'app-popups',
  templateUrl: './popups.component.html',
  styleUrls: ['./popups.component.scss']
})
export class PopupsComponent implements OnInit {

  @ViewChild('autoSwal', {static: true}) private swalComponent: SwalComponent;

  constructor() { }

  ngOnInit() {
  }

  autoCloseSwal() {
    setTimeout(() => {
      this.swalComponent.nativeSwal.close()      
    }, 2000);
    
  }

}
