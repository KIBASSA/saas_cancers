import { Component, OnInit } from '@angular/core';
import {Subscription} from 'rxjs/Subscription';
import { DomSanitizer } from '@angular/platform-browser';
import {Router} from '@angular/router';
import {PatientsApiService} from '../../patients/patient.service';
import {PatientsProviders} from '../../patients/patients.tools'
import {Patient, PatientToDiagnose, DiagnosticStatus, DiagnosedImage} from '../../patients/patient.model';

@Component({
  selector: 'app-start',
  templateUrl: './start.component.html',
  styleUrls: ['./start.component.scss']
})

export class StartComponent implements OnInit {

  patientsSubscription: Subscription;
  patientList : PatientToDiagnose[]=[];
  allChecked:boolean
  globalDiagnosticStatus : DiagnosticStatus
  constructor(private patientsApi: PatientsApiService, private patientsProviders:PatientsProviders, private sanitizer: DomSanitizer, private route:Router) {}

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
    this.globalDiagnosticStatus = DiagnosticStatus.Start
    var patientIndex = 0
    this.patientList.forEach(patient=>{
      patient.diagnosticStatus = DiagnosticStatus.Start
      this.patientsApi.predict_cancer(patient.cancerImages.toString()).subscribe(res=> 
        {
          patientIndex += 1
          var data = JSON.parse(res)
          data.forEach((element: any) => {
            patient.diagnosedImages.forEach(diagnosedImage=> 
            {
              diagnosedImage.withCancer =  element[diagnosedImage.url] == "cancer"
            })
            patient.hasCancer = patient.diagnosedImages.some(u => u.withCancer);
          });
          patient.diagnosticStatus = DiagnosticStatus.Finished
          patient.isDiagnosed = true
          if (patientIndex == this.patientList.length)
              this.globalDiagnosticStatus = DiagnosticStatus.Finished
        })
    })
  }

  diagnosticStatusStart(input:any)
  {
    if(input instanceof PatientToDiagnose){ 
      return (input as PatientToDiagnose).diagnosticStatus == DiagnosticStatus.Start
    }
    else
    {
      return (input as DiagnosticStatus) == DiagnosticStatus.Start
    }
  }

  diagnosticStatusFinished(input:any)
  {
    if(input instanceof PatientToDiagnose){ 
      return (input as PatientToDiagnose).diagnosticStatus == DiagnosticStatus.Finished
    }
    else
    {
      return (input as DiagnosticStatus) == DiagnosticStatus.Finished
    }
  }

  getImageStyle(patient: PatientToDiagnose,   image:DiagnosedImage)
  {
    if (!this.diagnosticStatusFinished(patient))
        return this.sanitizer.bypassSecurityTrustStyle("")

    if (image.withCancer)
      return this.sanitizer.bypassSecurityTrustStyle("border: 3px dashed var(--pink)")
    
    return this.sanitizer.bypassSecurityTrustStyle("border: 3px dashed var(--green)")
  }

  onSave()
  {
    this.patientsApi.update_patients_as_diagnosed(this.patientList).subscribe(res=> 
      {
        console.log("The diagnosis has been saved")
        this.route.navigate(['/patients/diagnosed']);
      });
  }

  ngOnDestroy(): void {
    this.patientsSubscription.unsubscribe();
  }
}

