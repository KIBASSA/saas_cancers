import {Patient} from './patient.model';
export class PatientFactory {
    public getPatient(item): Patient {
        console.log("item :" + typeof(item))
        return new Patient(item["id"], 
                                item["name"],
                                  item["email"],
                                    item["image"], 
                                        item["is_diagnosed"], 
                                        item["has_cancer"],
                                        new Date(item["registration_date"]["year"], item["registration_date"]["month"], item["registration_date"]["day"]),
                                        new Date(item["diagnosis_date"]["year"], item["diagnosis_date"]["month"], item["diagnosis_date"]["day"]),
                                        item["cancer_images"]);
    }
  }
  
  export class PatientsProviders {
    public getPatients(data): Patient[] {
        var items : Patient[] = [];
        let factory = new PatientFactory()
        data.forEach(function (value) {
            let item = JSON.parse(value);
            let patient = factory.getPatient(item)
            items.push(patient);
        });
    return items
  }
}