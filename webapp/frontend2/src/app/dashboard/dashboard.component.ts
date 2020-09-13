import { Component, OnInit, OnDestroy } from '@angular/core';
import {Subscription} from 'rxjs/Subscription';
import {PatientsApiService} from '../patients/patient.service';
import {PatientsProviders} from '../patients/patients.tools'
import {Patient} from '../patients/patient.model';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})

export class DashboardComponent implements OnInit, OnDestroy  {

  undiagnosedPatientsSubscription: Subscription;
  diagnosedPatientsSubscription: Subscription;
  undiagnosedPatientList : Patient[];
  diagnosedPatientList: Patient[];

  constructor(private patientsApi: PatientsApiService, private patientsProviders:PatientsProviders) {}

  
  ngOnInit() 
  {
      this.undiagnosedPatientsSubscription = this.patientsApi
                                              .getUndiagnosedPatients()
      .subscribe(res => {
                  this.undiagnosedPatientList = this.patientsProviders.getPatients(res).filter((u, i) => i < 4);
        },
        console.error
      );


      this.diagnosedPatientsSubscription = this.patientsApi
                                              .getDiagnosedPatients()
      .subscribe(res => {
                  this.diagnosedPatientList = this.patientsProviders.getPatients(res).filter((u, i) => i < 4);
        },
        console.error
      );

  }

 
  

  ngOnDestroy() {
    this.undiagnosedPatientsSubscription.unsubscribe();
    this.diagnosedPatientsSubscription.unsubscribe();
  }
}
