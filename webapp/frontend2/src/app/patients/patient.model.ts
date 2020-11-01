
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
