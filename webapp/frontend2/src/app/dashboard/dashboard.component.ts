import { Component, OnInit, OnDestroy } from '@angular/core';
import {Subscription} from 'rxjs/Subscription';
import {PatientsApiService} from '../patients/patient.service';
import {Patient} from '../patients/patient.model';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})

export class DashboardComponent implements OnInit, OnDestroy  {

  undiagnosedPatients: Subscription;
  undiagnosedPatientList : Patient[];

  constructor(private patientsApi: PatientsApiService) {}

  ngOnInit() 
  {
      this.undiagnosedPatients = this.patientsApi
                                              .getUndiagnosedPatients()
      .subscribe(res => {
          this.undiagnosedPatientList = res;
          console.log(this.undiagnosedPatientList)
        },
        console.error
      );
  }

  ngOnDestroy() {
    this.undiagnosedPatients.unsubscribe();
  }
}
