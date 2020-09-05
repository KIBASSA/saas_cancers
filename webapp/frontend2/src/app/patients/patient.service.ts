import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import { Observable, Subject, asapScheduler, pipe, of, from, interval, merge, fromEvent } from 'rxjs';
import 'rxjs/add/operator/catch';
import {API_URL} from '../../environments/environment';
import {Patient} from './patient.model';
import { catchError, retry } from 'rxjs/operators';
@Injectable()
export class PatientsApiService {

  constructor(private http: HttpClient) {
  }

  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }

  // GET list of public, future events
  getUndiagnosedPatients(): Observable<Patient[]> {
    return this.http
      .get(`${API_URL}/undiagnosed_patients`).pipe(
        retry(3),
        catchError(err => {
                 console.log(err);
                 return of(null);
        }));
  }
}
