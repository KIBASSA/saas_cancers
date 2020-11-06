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

  getBase64Image(imgURL) {
    fetch(imgURL)
      .then(response => response.text())
      .then(contents => console.log(contents))
      .catch(() => console.log("Can’t access " + imgURL + " response. Blocked by browser?"))

    //var xhr = new XMLHttpRequest();       
    //xhr.open("GET", imgURL, true); 
    //xhr.responseType = "blob";
    //xhr.onload = function (e) {
    //        console.log(this.response);
    //        var reader = new FileReader();
    //        reader.onload = function(event) {
    //           var res = event.target.result;
    //           console.log(res)
    //        }
    //        var file = this.response;
    //        reader.readAsDataURL(file)
    //};
    //xhr.send()
    
    
    //var dataURL = canvas.toDataURL("image/png");
    //return dataURL.replace(/^data:image\/(png|jpg);base64,/, "");
    return "salut"
  }

  
  onLaunch(eve: any)
  {
    this.patientList.forEach(item => 
      {
        item.cancerImages.forEach(image=> {
          console.log(this.getBase64Image(image))
        });
      });
  }

  ngOnDestroy(): void {
    this.patientsSubscription.unsubscribe();
  }
}
