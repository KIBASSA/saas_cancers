import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes,  } from '@angular/router';
import { AnnotateComponent } from "./annotate/annotate.component"

const routes: Routes = [
    { path: 'annotate', component: AnnotateComponent }
  ];
  
  
@NgModule({
    declarations: [AnnotateComponent],
    imports: [
      CommonModule,
      FormsModule,
      RouterModule.forChild(routes)
    ]
  })
  export class AnnotationsModule { }
  