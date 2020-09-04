import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LoginComponent } from './login/login.component';
import { Login2Component } from './login2/login2.component';
import { RegisterComponent } from './register/register.component';
import { Register2Component } from './register2/register2.component';
import { LockscreenComponent } from './lockscreen/lockscreen.component';
import { Routes, RouterModule } from '@angular/router';

const routes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: 'login-2', component: Login2Component },
  { path: 'register', component: RegisterComponent },
  { path: 'register-2', component: Register2Component },
  { path: 'lock-screen', component: LockscreenComponent }
]

@NgModule({
  declarations: [LoginComponent, Login2Component, RegisterComponent, Register2Component, LockscreenComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes)
  ]
})
export class UserPagesModule { }
