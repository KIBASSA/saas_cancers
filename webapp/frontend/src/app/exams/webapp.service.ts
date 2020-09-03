import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import { Observable, Subject, asapScheduler, pipe, of, from, interval, merge, fromEvent } from 'rxjs';
import 'rxjs/add/operator/catch';
import {API_URL} from '../../environments/environment';
import {Exam} from './exam.model';
import { catchError, retry } from 'rxjs/operators';
@Injectable()
export class ExamsApiService {

  constructor(private http: HttpClient) {
  }

  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }

  // GET list of public, future events
  getExams(): Observable<Exam[]> {
    return this.http
      .get(`${API_URL}/exams`).pipe(
        retry(3),
        catchError(err => {
                 console.log(err);
                 return of(null);
        }));
  }
}
