import { Component, OnInit, OnDestroy, ViewChild  } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import {Subscription} from 'rxjs/Subscription';
import { DropzoneComponent, DropzoneDirective } from 'ngx-dropzone-wrapper';
import {PatientsApiService} from '../patient.service';
import {PatientFactory} from '../patients.tools'
import { Patient } from '../patient.model';
@Component({
  selector: 'app-item',
  templateUrl: './item.component.html',
  styleUrls: ['./item.component.scss']
})
export class ItemComponent implements OnInit, OnDestroy {
  patient: Patient;
  private sub: any;
  submitted = false;
  cancerImagesSubscription: Subscription;
  uploadedImages:string[]

  @ViewChild(DropzoneComponent, { static: false }) componentRef?: DropzoneComponent;
  @ViewChild(DropzoneDirective, { static: false }) directiveRef?: DropzoneDirective;

  constructor(private route: ActivatedRoute, private patientsApi: PatientsApiService, private patientFactory:PatientFactory) { }
  
  ngOnInit() 
  {
    this.uploadedImages = []
    this.sub = this.route.params.subscribe(params => {
      this.patientsApi.getPatientById(params['id']).subscribe(res => {
            let item = JSON.parse(res);
            this.patient = this.patientFactory.getPatient(item)
            console.log(this.patient)
        },
      console.error
      );
   });
  }

  onSubmit() {
    if (this.uploadedImages.length == 0)
        return;

    this.submitted = true;
    console.log("onSubmit....")
    console.log(typeof this.patient.id)
    console.log(typeof this.uploadedImages)
    this.cancerImagesSubscription = this.patientsApi
                                      .addCancerImages2(this.patient.id, this.uploadedImages)
                                      .subscribe(res => {
                                        console.log("bug")
                                          if (res != null)
                                          {
                                            console.log("patient " + res.name + " created!")
                                            // replace current patient with new one
                                            let item = JSON.parse(res);
                                            this.patient = this.patientFactory.getPatient(item)
                                            // reset dropzone component
                                            this.componentRef.directiveRef.reset();
                                            this.submitted = false;
                                          }
                                        },
                                        console.error
                                        );
  }

  onSuccess(data: any) {
    this.uploadedImages.push(data[0]["dataURL"])
    console.log(" this.uploadedImages : " +  this.uploadedImages.length)
  }
  onSending(data:any)
  {
    //this.uploadedImages.push(data[0]["dataURL"])
  }
  ngOnDestroy(): void {
    this.sub.unsubscribe();
  }
}