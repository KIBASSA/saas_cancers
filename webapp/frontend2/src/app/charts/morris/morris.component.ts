import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-morris',
  templateUrl: './morris.component.html',
  styleUrls: ['./morris.component.scss']
})
export class MorrisComponent implements OnInit {
  chartLineData = [
    {
    y: '2006',
    a: 100,
    b: 150 
    },
    {
      y: '2007',
      a: 75,
      b: 65
    },
    {
      y: '2008',
      a: 50,
      b: 40
    },
    {
      y: '2009',
      a: 75,
      b: 65
    },
    {
      y: '2010',
      a: 50,
      b: 40
    },
    {
      y: '2011',
      a: 75,
      b: 65
    },
    {
      y: '2012',
      a: 100,
      b: 90
    }
  ];

  public chartLineOptions = {
    xkey : 'y',
    ykeys : ['a', 'b'],
    labels : ['Series A', 'Series B'],
    lineColors : ['#63CF72', '#F36368', '#76C1FA', '#FABA66']
  }

  chartBarData = [
    {
    y: '2006',
    a: 100,
    b: 90
    },
    {
      y: '2007',
      a: 75,
      b: 65
    },
    {
      y: '2008',
      a: 50,
      b: 40
    },
    {
      y: '2009',
      a: 75,
      b: 65
    },
    {
      y: '2010',
      a: 50,
      b: 40
    },
    {
      y: '2011',
      a: 75,
      b: 65
    },
    {
      y: '2012',
      a: 100,
      b: 90
    }
  ];

  public chartBarOptions = {
    xkey : 'y',
    ykeys : ['a', 'b'],
    labels : ['Series A', 'Series B'],
    barColors : ['#63CF72', '#F36368', '#76C1FA', '#FABA66']
  }

  chartAreaData = [
    {
    y: '2006',
    a: 100,
    b: 90
    },
    {
      y: '2007',
      a: 75,
      b: 105
    },
    {
      y: '2008',
      a: 50,
      b: 40
    },
    {
      y: '2009',
      a: 75,
      b: 65
    },
    {
      y: '2010',
      a: 50,
      b: 40
    },
    {
      y: '2011',
      a: 75,
      b: 65
    },
    {
      y: '2012',
      a: 100,
      b: 90
    }
  ];

  public chartAreaOptions = {
    xkey : 'y',
    ykeys : ['a', 'b'],
    labels : ['Series A', 'Series B'],
    lineColors : ['#76C1FA', '#F36368', '#63CF72', '#FABA66']
  }

  chartDonutData = [{
    label: "Download Sales",
    value: 12
    },
    {
      label: "In-Store Sales",
      value: 30
    },
    {
      label: "Mail-Order Sales",
      value: 20
    }
  ];

  public chartDonutOptions = {
    colors : ['#76C1FA', '#F36368', '#63CF72', '#FABA66']
  }
  
  constructor() { }

  ngOnInit() {
  }

}
