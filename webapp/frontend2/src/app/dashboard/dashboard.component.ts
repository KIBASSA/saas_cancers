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
          var items : Patient[] = [];
          res.forEach(function (value) {
            let item = JSON.parse(value);
            let patient = new Patient(item["id"], 
                                        item["name"], 
                                            item["image"], 
                                                item["is_diagnosed"], 
                                                  item["has_cancer"],
                                                    item["registration_date"],
                                                        item["diagnosis_date"])
            items.push(patient)
          }); 
          this.undiagnosedPatientList = items;
        },
        console.error
      );
  }

  ngOnDestroy() {
    this.undiagnosedPatients.unsubscribe();
  }
}
