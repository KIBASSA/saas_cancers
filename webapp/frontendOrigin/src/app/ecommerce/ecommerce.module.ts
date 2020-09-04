import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { InvoiceComponent } from './invoice/invoice.component';
import { Routes, RouterModule } from '@angular/router';
import { PricingComponent } from './pricing/pricing.component';
import { OrdersComponent } from './orders/orders.component';
import { DataTablesModule } from 'angular-datatables';

const routes: Routes = [
  { path: 'invoice', component: InvoiceComponent },
  { path: 'pricing', component: PricingComponent },
  { path: 'orders', component: OrdersComponent },
]

@NgModule({
  declarations: [InvoiceComponent, PricingComponent, OrdersComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
    DataTablesModule
  ]
})
export class EcommerceModule { }
