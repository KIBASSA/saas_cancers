import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { Faq2Component } from './faq2.component';

describe('Faq2Component', () => {
  let component: Faq2Component;
  let fixture: ComponentFixture<Faq2Component>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ Faq2Component ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(Faq2Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
