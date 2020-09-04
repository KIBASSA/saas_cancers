import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BlankPageComponent } from './blank-page/blank-page.component';
import { ProfileComponent } from './profile/profile.component';
import { FaqComponent } from './faq/faq.component';
import { Faq2Component } from './faq2/faq2.component';
import { NewsGridComponent } from './news-grid/news-grid.component';
import { TimelineComponent } from './timeline/timeline.component';
import { SearchResultsComponent } from './search-results/search-results.component';
import { PortfolioComponent } from './portfolio/portfolio.component';
import { Routes, RouterModule } from '@angular/router';
import { BarRatingModule } from "ngx-bar-rating";
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';

const routes : Routes = [
  { path: 'blank-page', component: BlankPageComponent },
  { path: 'profile', component: ProfileComponent },
  { path: 'faq', component: FaqComponent },
  { path: 'faq-2', component: Faq2Component },
  { path: 'news-grid', component: NewsGridComponent },
  { path: 'timeline', component: TimelineComponent },
  { path: 'search-results', component: SearchResultsComponent },
  { path: 'portfolio', component: PortfolioComponent },
]

@NgModule({
  declarations: [BlankPageComponent, ProfileComponent, FaqComponent, Faq2Component, NewsGridComponent, TimelineComponent, SearchResultsComponent, PortfolioComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(routes),
    BarRatingModule,
    NgbModule
  ]
})
export class GeneralPagesModule { }
