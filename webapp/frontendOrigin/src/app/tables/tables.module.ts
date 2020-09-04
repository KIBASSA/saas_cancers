import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BasicTableComponent } from './basic-table/basic-table.component';
import { RouterModule, Routes } from '@angular/router';
import { DataTableComponent } from './data-table/data-table.component';
import { DataTablesModule } from 'angular-datatables';

const routes: Routes = [
  { path: 'basic-table', component: BasicTableComponent },
  { path: 'data-table', component: DataTableComponent },
]

@NgModule({
  declarations: [BasicTableComponent, DataTableComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
    DataTablesModule
  ]
})
export class TablesModule { }
