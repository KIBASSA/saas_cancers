
export class Patient 
{
  registrationDateStr: string;
  diagnosisDateStr : string;

  constructor(
    public id:string,
    public name:string,
    public image:string,
    public isDiagnosed?:boolean,
    public hasCancer?:boolean,
    public registrationDate?:Date,
    public diagnosisDate?:Date,
    public cancerImages?:string[]
  )
  {
    this.id = id
    this.name = name
    this.image = image
    this.isDiagnosed = isDiagnosed
    this.hasCancer = hasCancer
    this.registrationDate = registrationDate
    this.diagnosisDate = diagnosisDate
    this.cancerImages = cancerImages
    
    if (registrationDate != undefined)
        this.registrationDateStr = registrationDate.toDateString()
    
    if (diagnosisDate != undefined)
        this.diagnosisDateStr = diagnosisDate.toDateString()
  }
}

export enum DiagnosticStatus {
  Start,
  Finished
}


export class PatientToDiagnose extends Patient
{
  checked : boolean
  diagnosticStatus:DiagnosticStatus;
  diagnosedImages: DiagnosedImage[] = [];
  constructor(public patient:Patient)
  {
      super(patient.id, patient.name,patient.image,patient.isDiagnosed,patient.hasCancer,patient.registrationDate, patient.diagnosisDate, patient.cancerImages);
      patient.cancerImages.forEach(image=> 
        {
          var data = new DiagnosedImage();
          data.url = image;
          this.diagnosedImages.push(data);
        })
  }
}

export class DiagnosedImage
{
  url : string;
  withCancer:boolean;
}