from pymongo import MongoClient
from pymongo import errors
import pymongo 
from datetime import date
from models import Patient, Doctor
from bson.objectid import ObjectId

class CancerDBAPI:
    def __init__(self, connectionstring="mongodb://admincancer:2563crRT8@localhost:27017/"):
        self.conn = MongoClient(connectionstring)
        self.db = self.conn.CancerDB

        self.collection_cancers = self.db.cancers
        self.collection_patients = self.db.patients
        self.collection_models = self.db.models
        self.collection_users = self.db.users
        self.collection_doctors = self.db.doctors

        
    
    def add_cancers(self,external_ids, types, names, imagelinks):
        bulk_to_insert = []
        assert len(external_ids) == len(types) == len(names) == len(imagelinks)

        for index, external_id in enumerate(external_ids):
            bulk_to_insert.append(
                {
                    "external_id":external_id,
                    "type":types[index],
                    "name":names[index],
                    "imagelink":imagelinks[index]
                })
        try:
             self.collection_cancers.insert_many(bulk_to_insert)
        except Exception as e:
            print(e)

    def patient_awaiting_diagnosis(self):
        """
        This method returns the list of undiagnosed patients who have images of their cancer. 
        """
        patients = []
        try:
            result = self.collection_patients.find({"is_diagnosed":False}).sort("registration_date",pymongo.DESCENDING)
            for item in result:
                cancer_images = self._get_cancer_images(item)
                if len(cancer_images):
                    patient = self._get_patient(item)
                    patients.append(patient)
        except Exception as e:
            raise Exception(e)
        
        return patients

    def get_diagnosed_patients(self, is_diagnosed):
        patients = []
        try:
            result = self.collection_patients.find({"is_diagnosed":is_diagnosed}).sort("diagnosis_date",pymongo.DESCENDING)
            for item in result:
                patient = self._get_patient(item)
                patients.append(patient)
        except Exception as e:
            raise Exception(e)
        
        return patients
    

    def _get_patient(self,mongo_patient):
        cancer_images = self._get_cancer_images(mongo_patient)
        return Patient(id=str(mongo_patient["_id"]),
                                    name=mongo_patient["name"],
                                      email = mongo_patient["email"],
                                        image=mongo_patient["image"],
                                            is_diagnosed=mongo_patient["is_diagnosed"], 
                                                has_cancer=mongo_patient["has_cancer"],
                                                  registration_date=mongo_patient["registration_date"],
                                                   diagnosis_date=mongo_patient["diagnosis_date"],
                                                   cancer_images=cancer_images)

    def _get_json_patient(self, patient_model):
        data = {}

        if patient_model.id:
            data["id"] = patient_model.id

        data["name"] = patient_model.name

        data["email"] = patient_model.email
        
        data["image"] = patient_model.image
        
        if patient_model.is_diagnosed:
            data["is_diagnosed"] = patient_model.is_diagnosed

        if patient_model.has_cancer:
            data["has_cancer"] = patient_model.has_cancer

        if patient_model.registration_date:
            data["registration_date"] = patient_model.registration_date
        
        if patient_model.diagnosis_date:
            data["diagnosis_date"] = patient_model.diagnosis_date

        return data


    def get_all_patients(self):
        patients = []
        try:
            result = self.collection_patients.find({}).sort("registration_date",pymongo.DESCENDING)
            for item in result:
                patient = self._get_patient(item)
                patients.append(patient)
        except Exception as e:
            raise Exception(e)
        
        return patients
    
    def insert_patient(self, patient):

        ## add user entity
        user_count = self.collection_users.count_documents({'email': patient.email})
        if user_count == 0:
            print("youpiiii")
            data = {}
            data["email"] = patient.email
            data["password"] = "mypass"
            data["type"] = "patient"
            self.collection_users.insert_one(data)

        ## add patient entity
        item_count = self.collection_patients.count_documents({'email': patient.email})
        if item_count == 0:
            json_patient = self._get_json_patient(patient)
            return self.collection_patients.insert_one(json_patient).inserted_id
        
        return self.get_patient_by_name(patient.email)

    def get_patient_by_id(self, patient_id):
        result = list(self.collection_patients.find({'_id':ObjectId(patient_id)}))
        if len(result) == 0:
            return None
        return self._get_patient(result[0])
    
    def get_patient_by_name(self, name):
        return self.collection_patients.find({'name':name})

    def update_patient(self,patient):
        #print("patient.is_diagnosed :", patient.is_diagnosed)
        #print("patient.id :", patient.id)
        self.collection_patients.update_one({"_id": ObjectId(patient.id)}, 
                                               {"$set":
                                                       {"image": patient.image,
                                                        "diagnosis_date": patient.diagnosis_date,
                                                       "is_diagnosed": patient.is_diagnosed,
                                                       "has_cancer": patient.has_cancer}})

    def _get_cancer_images(self, mongo_patient_data):
        original_images = []
        if "cancers" in mongo_patient_data:
            if "breast" in mongo_patient_data["cancers"]:
                if "images" in mongo_patient_data["cancers"]["breast"]:
                    original_images = mongo_patient_data["cancers"]["breast"]["images"]
        return original_images

    def insert_cancer_images(self, patient_id,images):
        item_count = self.collection_patients.count_documents({'_id':ObjectId(patient_id)})

        if item_count == 0:
            raise Exception("The targeted patient does not exist ")
        
        result = list(self.collection_patients.find({'_id':ObjectId(patient_id)}))[0]
        original_images = self._get_cancer_images(result)
        images.extend(original_images)
        print("images after extend : ", images)           

        self.collection_patients.update_one({"_id": ObjectId(patient_id)}, 
                                               {"$set":
                                                       {"cancers": 
                                                            {
                                                                "breast":
                                                                {
                                                                    "images":images
                                                                }
                                                            }
                                                       }
                                                })
        print("patient updated")

    def add_model_accuracy(self, model_name, accuracy):
        item_count = self.collection_models.count_documents({'model_name':model_name})
        
        model_id = ""
        if item_count == 0:
            model_id = self.collection_models.insert_one({"model_name": model_name}).inserted_id
        
        result = list(self.collection_models.find({'model_name':model_name}))[0]
    

    def login(self, email, password):
        
        result = list(self.collection_users.find({'email':email, 'password':password}))
        if len(result) == 0:
            return None

        if result[0]["type"] == "doctor":
            doctor_bd = list(self.collection_doctors.find({'email':email}))
            if len(doctor_bd) == 0:
                raise Exception("The user exists but the link with the doctor entity does not exist")
            doctor_bd = doctor_bd[0]
            return Doctor(doctor_bd["_id"],doctor_bd["email"], doctor_bd["name"], doctor_bd["image"],doctor_bd["roles"])
        elif result[0]["type"] == "patient":
            patient_bd = list(self.collection_patients.find({'email':email}))
            if len(patient_bd) == 0:
                raise Exception("The user exists but the link with the patient entity does not exist")
            patient_bd = patient_bd[0]
            return self.__get_patient(patient_bd)
        
        raise Exception("Type {0} has no specific table".format(result[0]["type"]))

