import { Component, OnInit } from '@angular/core';
import { Select2OptionData } from 'ng2-select2';
import {Observable} from 'rxjs';
import {debounceTime, distinctUntilChanged, map} from 'rxjs/operators';

const states = ['Alabama', 'Alaska', 'American Samoa', 'Arizona', 'Arkansas', 'California', 'Colorado',
  'Connecticut', 'Delaware', 'District Of Columbia', 'Federated States Of Micronesia', 'Florida', 'Georgia',
  'Guam', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
  'Marshall Islands', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana',
  'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
  'Northern Mariana Islands', 'Ohio', 'Oklahoma', 'Oregon', 'Palau', 'Pennsylvania', 'Puerto Rico', 'Rhode Island',
  'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virgin Islands', 'Virginia',
  'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'];

@Component({
  selector: 'app-basic-elements',
  templateUrl: './basic-elements.component.html',
  styleUrls: ['./basic-elements.component.scss']
})
export class BasicElementsComponent implements OnInit {

  public select2ExampleData: Array<Select2OptionData>;
  singleSelect2Options: Select2Options;
  multipleSelect2Options: Select2Options;

  constructor() { }

  ngOnInit() {
    this.select2ExampleData = [
      {
        id: 'AL',
        text: 'Alabama'
      },
      {
        id: 'WY',
        text: 'Wyoming'
      },
      {
        id: 'AM',
        text: 'America'
      },
      {
        id: 'CA',
        text: 'Canada'
      }
    ];
    this.singleSelect2Options = {
      width: '100%'
    }
    this.multipleSelect2Options = {
      width: '100%',
      multiple: true,
      placeholder: 'Select multiple items'
    }
  }

  typeaheadData = (text$: Observable<string>) =>
    text$.pipe(
      debounceTime(200),
      distinctUntilChanged(),
      map(term => term.length < 1 ? []
        : states.filter(v => v.toLowerCase().indexOf(term.toLowerCase()) > -1).slice(0, 10))
    );

}
