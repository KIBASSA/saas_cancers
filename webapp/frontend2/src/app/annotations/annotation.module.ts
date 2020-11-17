import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChartsModule } from 'ng2-charts';
import { MorrisJsModule } from 'angular-morris-js';
import { ChartistModule } from 'ng-chartist';
import { RouterModule, Routes,  } from '@angular/router';
import { AnnotateComponent } from "./annotate/annotate.component";
import { PerformanceComponent } from './performance/performance.component'

const routes: Routes = [
    { path: 'annotate', component: AnnotateComponent },
    { path: 'performance', component: PerformanceComponent}
  ];
  
  
@NgModule({
    declarations: [AnnotateComponent, PerformanceComponent],
    imports: [
      CommonModule,
      FormsModule,
      RouterModule.forChild(routes),
      ChartsModule,
      MorrisJsModule,
      ChartistModule
    ]
  })
  export class AnnotationsModule { }
  