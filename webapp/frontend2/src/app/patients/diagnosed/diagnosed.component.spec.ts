import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DiagnosedComponent } from './diagnosed.component';

describe('DiagnosedComponent', () => {
  let component: DiagnosedComponent;
  let fixture: ComponentFixture<DiagnosedComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DiagnosedComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DiagnosedComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
