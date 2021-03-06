import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import {Subscription} from 'rxjs/Subscription';
import {Router} from '@angular/router';

import {PatientsApiService} from '../patient.service';
import {Patient} from '../patient.model';
@Component({
  selector: 'app-new',
  templateUrl: './new.component.html',
  styleUrls: ['./new.component.scss']
})
export class NewComponent implements OnInit {

  patient:Patient
  uploadedImage:string
  registerForm: FormGroup;
  submitted = false;
  patientsSubscription: Subscription;
  loading:boolean

  constructor(private patientsApi: PatientsApiService, private formBuilder: FormBuilder, private route:Router) { }

  ngOnInit() {
    this.registerForm = this.formBuilder.group({
      firstName: ['', Validators.required],
      lastName: ['', Validators.required],
      email: ['', [Validators.required, Validators.email]]
    });
  }
  onSending(data: any): void {
    // data [ File , xhr, formData]
    const file = data[0];
    console.log(data[0])
    console.log(typeof(data[0]))
    //console.log(data[0]["dataURL"])
    console.log(data[0]["dataURL"])
    console.log(data[0]["previewElement"]["dataURL"])
  }

  // convenience getter for easy access to form fields
  get f() { return this.registerForm.controls; }

  onSubmit() {
    
    this.submitted = true;

    // stop here if form is invalid
    if (this.registerForm.invalid) {
        return;
    }
    // display form values on success
    let result  = this.registerForm.value //JSON.stringify(this.registerForm.value, null, 4)
    
    this.patient = new Patient("",result["firstName"] + " "  + result["lastName"], this.uploadedImage)
    this.loading = true
    this.patientsSubscription = this.patientsApi
                                      .addPatient(this.patient)
                                      .subscribe(res => {
                                          if (res != null)
                                          {
                                            console.log("patient " + res.name + " created!")
                                            this.route.navigate(['/patients/list']);
                                            this.loading = false
                                          }
                                        },
                                        console.error
                                        );
  }

  onReset() {
    this.submitted = false;
    this.registerForm.reset();
  }

  onSendingmultiple() {

  }
  
  onError() {
  
  }
  onSuccess(data: any) {
    console.log(data)
    console.log(data[0]["dataURL"])
    //this.patient.image = data[0]["dataURL"]
    this.uploadedImage = data[0]["dataURL"]
    console.log("finished")

    //fetch(data[0]["dataURL"])
    //.then(res => res.blob())
    //.then(console.log)
  }
}
