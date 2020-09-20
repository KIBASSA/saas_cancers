import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DropzoneModule } from 'ngx-dropzone-wrapper';
import { DROPZONE_CONFIG } from 'ngx-dropzone-wrapper';
import { DropzoneConfigInterface } from 'ngx-dropzone-wrapper';

import { RouterModule, Routes,  } from '@angular/router';
import { NewComponent } from './new/new.component';
import { ListComponent } from './list/list.component';

const routes: Routes = [
    { path: 'new', component: NewComponent },
    { path: 'list', component: ListComponent }
  ];
  
  
@NgModule({
    declarations: [NewComponent, ListComponent],
    imports: [
      CommonModule,
      DropzoneModule,
      RouterModule.forChild(routes)
    ]
  })
  export class PatientsModule { }
  