import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})

export class DashboardComponent implements OnInit {

  constructor() { }

  ngOnInit() {
  }

  date: Date = new Date();

  //  bar chart 
  barChartData = [{
    label: '# of Votes',
    data: [10, 19, 16, 15, 17, 13],
    borderWidth: 1,
    fill: false
  }];

  barChartLabels = ["", "", "", "", "", ""];

  barChartOptions = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }],
      xAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      }
    }
  };

  barChartColors = [
    {
      backgroundColor: [
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6'
      ],
      borderColor: [
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6'
      ]
    }
  ];
  // area chart
  areaChartData = [{
    label: '#',
    data: [8, 8, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 5, 5, 4, 3, 4, 5, 4, 5, 4, 4, 3, 4, 6, 5, 6, 5, 6, 7, 6, 7, 6, 6, 5, 4, 5, 4, 6, 5, 7, 7, 6, 5, 3, 5, 4, 5, 3, 5, 4, 5, 6, 4, 5, 4, 3, 5, 4, 5, 6, 7, 5, 4, 5, 6, 4, 3, 4, 5, 7, 4, 6, 7, 6, 5, 4, 3, 4, 6, 5, 7, 6, 7, 5, 7, 5, 2, 3],
    borderWidth: 1,
    fill: true
  }, {
    label: '#',
    data: [18, 18, 18, 17, 16, 15, 16, 15, 14, 13, 14, 13, 14, 15, 14, 13, 14, 16, 16, 15, 13, 14, 13, 14, 16, 15, 16, 15, 16, 15, 16, 15, 13, 14, 15, 14, 15, 14, 16, 15, 17, 17, 16, 15, 13, 15, 14, 13, 15, 14, 13, 12, 13, 12, 14, 13, 15, 14, 16, 17, 15, 14, 15, 16, 14, 13, 14, 15, 17, 14, 16, 17, 16, 15, 14, 14, 16, 15, 17, 16, 17, 15, 17, 11, 15, 14, 15, 16, 13],
    borderWidth: 1,
    fill: true
  }, {
    label: '#',
    data: [29, 27, 26, 27, 27, 28, 26, 27, 28, 27, 25, 26, 24, 25, 27, 26, 27, 25, 27, 28, 26, 28, 27, 29, 27, 28, 26, 26, 27, 26, 27, 25, 24, 23, 25, 27, 25, 26, 27, 29, 27, 26, 28, 29, 27, 30, 31, 31, 30, 31, 30, 31, 30, 31, 32, 31, 30, 28, 27, 29, 28, 26, 27, 28, 26, 28, 27, 29, 28, 27, 25, 26, 27, 26, 27, 25, 26, 27, 29, 28, 27, 28, 26, 27, 25, 23, 25, 26, 27],
    borderWidth: 1,
    fill: false
  }];

  areaChartLabels = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""];

  areaChartOptions = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }],
      xAxes: [{
        gridLines: {
          drawBorder: false,
          display: false,
        }
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      },
      line: {
        tension: 0
      }
    }
  };

  areaChartColors = [
    {
      borderColor: '#3f50f6',
      backgroundColor: '#3f50f6'
    }, {
      borderColor: '#bcc1f3',
      backgroundColor: '#bcc1f3'
    }, {
      borderColor: '#ffab2d',
    }
  ];

  areaChartData1 = [{
    label: '#',
    data: [12, 19, 3, 5, 2, 3],
    borderWidth: 1,
    fill: true
  }];

  areaChartLabels1 = ["", "", "", "", "", ""];

  areaChartOptions1 = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }],
      xAxes: [{
        gridLines: {
          drawBorder: false,
          display: false,
        }
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      }
    }
  };

  areaChartColors1 = [
    {
      borderColor: '#00cccd',
      backgroundColor: '#e2f8f8'
    }
  ];

  areaChartData2 = [{
    label: '#',
    data: [13, 2, 15, 3, 19, 12],
    borderWidth: 1,
    fill: true
  }];

  areaChartLabels2 = ["", "", "", "", "", ""];

  areaChartOptions2 = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }],
      xAxes: [{
        gridLines: {
          drawBorder: false,
          display: false,
        }
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      },
    }
  };

  areaChartColors2 = [
    {
      borderColor: '#00cccd',
      backgroundColor: '#e2f8f8'
    }
  ];

  areaChartData3 = [{
    label: '#',
    data: [13, 2, 15, 3, 19, 12],
    borderWidth: 1,
    fill: true
  }];

  areaChartLabels3 = ["", "", "", "", "", ""];

  areaChartOptions3 = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }],
      xAxes: [{
        gridLines: {
          drawBorder: false,
          display: false,
        }
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      },
    }
  };

  areaChartColors3 = [
    {
      borderColor: '#ffab2d',
      backgroundColor: '#ffed92'
    }
  ];

  areaChartData4 = [{
    label: '#',
    data: [2, 19, 13, 5, 12, 10],
    borderWidth: 1,
    fill: true
  }];

  areaChartLabels4 = ["", "", "", "", "", ""];

  areaChartOptions4 = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }],
      xAxes: [{
        gridLines: {
          drawBorder: false,
          display: false,
        }
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      },
    }
  };

  areaChartColors4 = [
    {
      borderColor: '#ffab2d',
      backgroundColor: '#ffed92'
    }
  ];

  areaChartData5 = [{
    label: '#',
    data: [12, 9, 13, 5, 12, 3],
    borderWidth: 1,
    fill: true
  }];

  areaChartLabels5 = ["", "", "", "", "", ""];

  areaChartOptions5 = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        }
      }],
      xAxes: [{
        gridLines: {
          drawBorder: false,
          display: false,
        }
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      },
    }
  };

  areaChartColors5 = [
    {
      borderColor: '#00cccd',
      backgroundColor: '#e2f8f8'
    }
  ];

  // stacked bar chart

  barStackData = [{
    label: '# of Votes',
    data: [14, 12, 9, 15, 10, 12, 10],
    borderWidth: 1,
    fill: false
  },{
    label: '#',
    data: [17, 17, 17, 17, 17, 17, 17],
    borderWidth: 1,
    fill: false
  }];

  barStackLabels = ["S", "M", "T", "W", "T", "F", "S"];

  barStackOptions = {
    scales: {
      yAxes: [{
        ticks: {
          display: false,
        },
        gridLines: {
          drawBorder: false,
          display: false
        },
      }],
      xAxes: [{
        gridLines: {
          drawBorder: false,
          display: false
        },
        stacked: true,
        barPercentage: 0.3
      }]

    },
    legend: {
      display: false
    },
    elements: {
      point: {
        radius: 0
      }
    }
  };

  barStackColors = [
    {
      backgroundColor: [
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6'
      ],
      borderColor: [
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6',
        '#3f50f6'
      ]
    },{
      backgroundColor: [
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6'
      ],
      borderColor: [
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6',
        '#e6e6e6'
      ]
    }
  ];
}
