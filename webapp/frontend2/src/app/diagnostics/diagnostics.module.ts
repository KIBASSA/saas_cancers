import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes,  } from '@angular/router';
import { StartComponent } from './start/start.component';

const routes: Routes = [
    { path: 'start', component: StartComponent }
  ];
  
  
@NgModule({
    declarations: [StartComponent],
    imports: [
      CommonModule,
      FormsModule,
      RouterModule.forChild(routes)
    ]
  })
  export class DiagnosticsModule { }
  