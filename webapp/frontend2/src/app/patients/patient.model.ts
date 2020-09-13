
export class Patient 
{
  constructor(
    public id:string,
    public name:string,
    public image:string,
    public isDiagnosed:boolean,
    public hasCancer:boolean,
    public registrationDate:Date,
    public diagnosisDate:Date
  )
  {}
}
