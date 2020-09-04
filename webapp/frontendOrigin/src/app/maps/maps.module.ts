import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GoogleMapComponent } from './google-map/google-map.component';
import { Routes, RouterModule } from '@angular/router';
import { AgmCoreModule } from '@agm/core';

const routes: Routes = [
  { path: 'google-map', component : GoogleMapComponent },
]

@NgModule({
  declarations: [GoogleMapComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
    AgmCoreModule.forRoot({ apiKey: 'AIzaSyCnT63XUjqjPgXZ0lFTU_pdpfUX7swzTTM' })
  ]
})
export class MapsModule { }
