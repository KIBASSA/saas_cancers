import { Component, OnInit } from '@angular/core';
import {Subscription} from 'rxjs/Subscription';

import {PatientsApiService} from '../../patients/patient.service';
import {PatientsProviders} from '../../patients/patients.tools'
import {Patient} from '../../patients/patient.model';

@Component({
  selector: 'app-start',
  templateUrl: './start.component.html',
  styleUrls: ['./start.component.scss']
})
export class StartComponent implements OnInit {

  patientsSubscription: Subscription;
  patientList : Patient[];
  constructor(private patientsApi: PatientsApiService, private patientsProviders:PatientsProviders) {}

  ngOnInit() {
    this.patientsSubscription = this.patientsApi
                                              .getPatientAwaitingDiagnosis()
      .subscribe(res => {
                  this.patientList = this.patientsProviders.getPatients(res).filter((u, i) => i < 10);
        },
        console.error
      );
  }

  ngOnDestroy(): void {
    this.patientsSubscription.unsubscribe();
  }

}
