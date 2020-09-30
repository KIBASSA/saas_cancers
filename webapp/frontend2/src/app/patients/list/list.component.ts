import { Component, OnInit } from '@angular/core';

import {Subscription} from 'rxjs/Subscription';
import {PatientsApiService} from '../patient.service';
import {PatientsProviders} from '../patients.tools'
import {Patient} from '../patient.model';

@Component({
  selector: 'app-list',
  templateUrl: './list.component.html',
  styleUrls: ['./list.component.scss']
})
export class ListComponent implements OnInit {

  patientsSubscription: Subscription;
  patientList : Patient[];
  constructor(private patientsApi: PatientsApiService, private patientsProviders:PatientsProviders) {}

  ngOnInit() 
  {
    this.patientsSubscription = this.patientsApi
                                              .getAllPatients()
      .subscribe(res => {
                  this.patientList = this.patientsProviders.getPatients(res).filter((u, i) => i < 10);
        },
        console.error
      );
  }
}
