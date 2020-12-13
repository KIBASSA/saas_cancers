import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { LoginComponent } from "./login/login.component"
//import { RtlComponent } from './rtl/rtl.component';

import { AuthGuard } from './_helpers/auth.guard';
const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full', canActivate: [AuthGuard]},
  { path: 'dashboard', component: DashboardComponent , canActivate: [AuthGuard] },
  { path: 'login', component: LoginComponent },
  { path: 'diagnostics', loadChildren: () => import('./diagnostics/diagnostics.module').then(m => m.DiagnosticsModule) },
  { path: 'patients', loadChildren: () => import('./patients/patients.module').then(m => m.PatientsModule) },
  { path: 'annotations', loadChildren: () => import('./annotations/annotation.module').then(m => m.AnnotationsModule) },
  { path: '', redirectTo: '/rtl', pathMatch: 'full' },
  //{ path: 'rtl', component: RtlComponent },
  //{ path: 'basic-ui', loadChildren: () => import('./basic-ui/basic-ui.module').then(m => m.BasicUiModule) },
  //{ path: 'advanced-ui', loadChildren: () => import('./advanced-ui/advanced-ui.module').then(m => m.AdvancedUiModule) },
  //{ path: 'charts', loadChildren: () => import('./charts/charts.module').then(m => m.ChartsDemoModule) },
  //{ path: 'forms', loadChildren: () => import('./forms/form.module').then(m => m.FormModule) },
  //{ path: 'editors', loadChildren: () => import('./editors/editors.module').then(m => m.EditorsModule) },
  //{ path: 'tables', loadChildren: () => import('./tables/tables.module').then(m => m.TablesModule) },
  //{ path: 'icons', loadChildren: () => import('./icons/icons.module').then(m => m.IconsModule) },
  //{ path: 'maps', loadChildren: () => import('./maps/maps.module').then(m => m.MapsModule) },
  //{ path: 'general-pages', loadChildren: () => import('./general-pages/general-pages.module').then(m => m.GeneralPagesModule) },
  //{ path: 'ecommerce', loadChildren: () => import('./ecommerce/ecommerce.module').then(m => m.EcommerceModule) },
  //{ path: 'apps', loadChildren: () => import('./apps/apps.module').then(m => m.AppsModule) },
  //{ path: 'user-pages', loadChildren: () => import('./user-pages/user-pages.module').then(m => m.UserPagesModule) },
  //{ path: 'error-pages', loadChildren: () => import('./error-pages/error-pages.module').then(m => m.ErrorPagesModule) },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
