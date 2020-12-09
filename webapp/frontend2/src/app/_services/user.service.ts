import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import {API_URL} from '../../environments/environment';
import { User } from '../_models/user';

@Injectable({ providedIn: 'root' })
export class UserService {
    constructor(private http: HttpClient) { }

    getAll() {
        return this.http.get<User[]>(`${API_URL}/users`);
    }

    register(user: User) {
        return this.http.post(`${API_URL}/users/register`, user);
    }

    delete(id: number) {
        return this.http.delete(`${API_URL}/users/${id}`);
    }
}