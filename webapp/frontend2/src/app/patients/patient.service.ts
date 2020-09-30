import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import { Headers, URLSearchParams  } from '@angular/http';
import { Observable, Subject, asapScheduler, pipe, of, from, interval, merge, fromEvent } from 'rxjs';
import 'rxjs/add/operator/catch';
import {API_URL} from '../../environments/environment';
import {Patient} from './patient.model';
import { catchError, retry } from 'rxjs/operators';
@Injectable()
export class PatientsApiService {
FormData
  constructor(private http: HttpClient) {}

  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }

  private headers = new Headers({'Content-Type': 'application/json'});
  // GET list of public, future events
  getUndiagnosedPatients(): Observable<string[]> {
    return this.http
      .get(`${API_URL}/undiagnosed_patients`).pipe(
        retry(3),
        catchError(err => {
                 console.log(err);
                 return of(null);
        }));
  }

   // GET list of public, future events
   getDiagnosedPatients(): Observable<string[]> {
    return this.http
      .get(`${API_URL}/diagnosed_patients`).pipe(
        retry(3),
        catchError(err => {
                 console.log(err);
                 return of(null);
        }));
  }

  // GET list of public, future events
  getAllPatients(): Observable<string[]> {
    return this.http
      .get(`${API_URL}/all_patients`).pipe(
        retry(3),
        catchError(err => {
                 console.log(err);
                 return of(null);
        }));
  }

  //Put new Patient
  addPatient(patient : Patient): Observable<Patient> {
    let urlSearchParams = new URLSearchParams();
    urlSearchParams.append('patient', "{name:'salut'}");
    const formData = new FormData();
    formData.append('ptient',  JSON.stringify(patient));
    //urlSearchParams.append('password', password);
    return this.http.post<any>(`${API_URL}/add_patient`, formData)
      .pipe(
        catchError(err => {
          console.log(err);
          return of(null);
            })
      );
  }
}
