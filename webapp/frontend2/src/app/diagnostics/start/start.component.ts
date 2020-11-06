import { Component, OnInit } from '@angular/core';
import {Subscription} from 'rxjs/Subscription';

import {PatientsApiService} from '../../patients/patient.service';
import {PatientsProviders} from '../../patients/patients.tools'
import {Patient, PatientToDiagnose} from '../../patients/patient.model';

@Component({
  selector: 'app-start',
  templateUrl: './start.component.html',
  styleUrls: ['./start.component.scss']
})
export class StartComponent implements OnInit {

  patientsSubscription: Subscription;
  patientList : PatientToDiagnose[];
  allChecked:boolean
  constructor(private patientsApi: PatientsApiService, private patientsProviders:PatientsProviders) {}

  ngOnInit() {
    this.patientsSubscription = this.patientsApi
                                              .getPatientAwaitingDiagnosis()
      .subscribe(res => {
                  this.patientList = this.patientsProviders.getPatients(res).map((a)=> new PatientToDiagnose(a));
                  this.patientList.forEach(item => 
                    {
                       item.checked = true;
                    });
                  this.allChecked = true;
        },
        console.error
      );
  }
  onToDiagnoseAllPatientChange(eve:any)
  {
    this.patientList.forEach(item => { item.checked = this.allChecked});
  }
  onToDiagnosePatientChange(eve: any) {
    this.allChecked = !this.patientList.some(u => !u.checked);
  }

  onLaunch(eve: any)
  {
    console.log("salut")
  }

  ngOnDestroy(): void {
    this.patientsSubscription.unsubscribe();
  }
}
