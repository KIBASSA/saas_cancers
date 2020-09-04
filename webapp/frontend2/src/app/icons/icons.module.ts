import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Routes } from '@angular/router';
import { MdiComponent } from './mdi/mdi.component';
import { FlagIconsComponent } from './flag-icons/flag-icons.component';
import { FontAwesomeComponent } from './font-awesome/font-awesome.component';
import { SimpleLineIconsComponent } from './simple-line-icons/simple-line-icons.component';
import { ThemifyComponent } from './themify/themify.component';

const routes : Routes = [
  { path: 'mdi', component: MdiComponent },
  { path: 'flag-icons', component: FlagIconsComponent },
  { path: 'font-awesome', component: FontAwesomeComponent },
  { path: 'simple-line-icons', component: SimpleLineIconsComponent },
  { path: 'themify', component: ThemifyComponent },
]

@NgModule({
  declarations: [MdiComponent, FlagIconsComponent, FontAwesomeComponent, SimpleLineIconsComponent, ThemifyComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
  ]
})
export class IconsModule { }
