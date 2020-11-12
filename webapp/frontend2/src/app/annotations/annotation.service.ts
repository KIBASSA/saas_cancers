import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Observable, Subject, asapScheduler, pipe, of, from, interval, merge, fromEvent } from 'rxjs';
import 'rxjs/add/operator/catch';
import {API_URL} from '../../environments/environment';
import { catchError, retry } from 'rxjs/operators';
@Injectable()
export class AnnotationsApiService {
FormData
  constructor(private http: HttpClient) {}

  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }
  // Get Patient by her id
  get_sampled_images(): Observable<any> {
    //urlSearchParams.append('password', password);
    return this.http.get<any>(`${API_URL}/get_sampled_images`)
      .pipe(
        catchError(err => {
          console.log(err);
          return of(null);
            })
      );
  }
}
