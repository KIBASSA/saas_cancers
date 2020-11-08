import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DropzoneModule } from 'ngx-dropzone-wrapper';
import { DROPZONE_CONFIG } from 'ngx-dropzone-wrapper';
import { DropzoneConfigInterface } from 'ngx-dropzone-wrapper';

import { RouterModule, Routes,  } from '@angular/router';
import { NewComponent } from './new/new.component';
import { ListComponent } from './list/list.component';


import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ItemComponent } from './item/item.component';
import { DiagnosedComponent } from './diagnosed/diagnosed.component';

const routes: Routes = [
    { path: 'new', component: NewComponent },
    { path: 'list', component: ListComponent },
    { path: 'diagnosed', component: DiagnosedComponent },
    { path: 'patient/:id', component: ItemComponent },
  ];

  const DEFAULT_DROPZONE_CONFIG: DropzoneConfigInterface = {
    // Change this to your upload POST address:
      url: 'https://httpbin.org/post',
      maxFilesize: 50,
      acceptedFiles: 'image/*'
  };
  
@NgModule({
    declarations: [NewComponent, ListComponent,DiagnosedComponent, ItemComponent, DiagnosedComponent],
    imports: [
      CommonModule,
      DropzoneModule,
      FormsModule,
      ReactiveFormsModule,
      RouterModule.forChild(routes)
    ],
    providers: [
      {
        provide: DROPZONE_CONFIG,
        useValue: DEFAULT_DROPZONE_CONFIG
      }
    ]
  })
  export class PatientsModule { }
  