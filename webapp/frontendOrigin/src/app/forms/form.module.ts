import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BasicElementsComponent } from './basic-elements/basic-elements.component';
import { RouterModule, Routes } from '@angular/router';
import { Select2Module } from 'ng2-select2';
import { BarRatingModule } from "ngx-bar-rating";
import { DropzoneModule } from 'ngx-dropzone-wrapper';
import { DROPZONE_CONFIG } from 'ngx-dropzone-wrapper';
import { DropzoneConfigInterface } from 'ngx-dropzone-wrapper';
import { ColorPickerModule } from 'ngx-color-picker';
import { AngularDateTimePickerModule } from 'angular2-datetimepicker';
import { AmazingTimePickerModule } from 'amazing-time-picker';
import { TagInputModule } from 'ngx-chips';
import { CustomFormsModule } from 'ng2-validation';
import { FormWizardModule } from 'angular2-wizard';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { AdvancedElementsComponent } from './advanced-elements/advanced-elements.component';
import { ValidationComponent } from './validation/validation.component';
import { WizardComponent } from './wizard/wizard.component';

const DEFAULT_DROPZONE_CONFIG: DropzoneConfigInterface = {
  // Change this to your upload POST address:
    url: 'https://httpbin.org/post',
    maxFilesize: 50,
    acceptedFiles: 'image/*'
};
const routes: Routes = [
  { path: 'basic-elements', component: BasicElementsComponent },
  { path: 'advanced-elements', component: AdvancedElementsComponent },
  { path: 'validation', component: ValidationComponent },
  { path: 'wizard', component: WizardComponent }
]

@NgModule({
  declarations: [BasicElementsComponent, AdvancedElementsComponent, ValidationComponent, WizardComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
    Select2Module,
    NgbModule,
    BarRatingModule,
    DropzoneModule,
    ColorPickerModule,
    AngularDateTimePickerModule,
    AmazingTimePickerModule,
    TagInputModule,
    CustomFormsModule,
    FormWizardModule
  ],
  providers: [
    {
      provide: DROPZONE_CONFIG,
      useValue: DEFAULT_DROPZONE_CONFIG
    }
  ]
})
export class FormModule { }
