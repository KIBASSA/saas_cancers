import { DatePipe } from '@angular/common';
export class Datepicker {
    _date: string;
    constructor(private datePipe: DatePipe) {}
  
    set date(value) {
      let date = new Date(value.year, value.month, value.year);
      this._date= this.datePipe.transform(date, 'yyyy-MM-dd');
    }
  
    get date() { 
        return this._date
    }
  }