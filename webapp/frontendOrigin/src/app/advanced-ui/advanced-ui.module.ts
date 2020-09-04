import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DragulaModule } from 'ng2-dragula';
import { ClipboardModule } from 'ngx-clipboard';
import { ContextMenuModule } from 'ngx-contextmenu';
import { NouisliderModule } from 'ng2-nouislider';
import { CarouselModule } from 'ngx-owl-carousel-o';
import { SweetAlert2Module } from '@sweetalert2/ngx-sweetalert2';

import { DragulaComponent } from './dragula/dragula.component';
import { RouterModule, Routes,  } from '@angular/router';
import { ClipboardComponent } from './clipboard/clipboard.component';
import { MyContextMenuComponent } from './context-menu/context-menu.component';
import { SliderComponent } from './slider/slider.component';
import { CarouselComponent } from './carousel/carousel.component';
import { LoadersComponent } from './loaders/loaders.component';
import { PopupsComponent } from './popups/popups.component';

const routes: Routes = [
  { path: 'dragula', component: DragulaComponent },
  { path: 'clipboard', component: ClipboardComponent },
  { path: 'context-menu', component: MyContextMenuComponent },
  { path: 'slider', component: SliderComponent },
  { path: 'carousel', component: CarouselComponent },
  { path: 'loaders', component: LoadersComponent },
  { path: 'popups', component: PopupsComponent },
];

@NgModule({
  declarations: [DragulaComponent, ClipboardComponent, MyContextMenuComponent, SliderComponent, CarouselComponent, LoadersComponent, PopupsComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
    DragulaModule,
    ClipboardModule,
    ContextMenuModule,
    NouisliderModule,
    CarouselModule,
    SweetAlert2Module.forRoot()
  ]
})
export class AdvancedUiModule { }
