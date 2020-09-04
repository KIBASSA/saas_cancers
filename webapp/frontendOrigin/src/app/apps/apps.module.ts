import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { FullCalendarModule } from 'ng-fullcalendar';

import { EmailComponent } from './email/email.component';
import { Routes, RouterModule } from '@angular/router';
import { CalendarComponent } from './calendar/calendar.component';
import { ReactiveFormsModule } from '@angular/forms';

const routes: Routes = [
  { path: 'email', component: EmailComponent },
  { path: 'calendar', component: CalendarComponent },
]

@NgModule({
  declarations: [EmailComponent, CalendarComponent ],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
    FullCalendarModule,
    ReactiveFormsModule,
  ]
})
export class AppsModule { }
