from pymongo import MongoClient
from pymongo import errors
import pymongo 
from datetime import date
from models import Patient
from bson.objectid import ObjectId

class CancerDBAPI:
    def __init__(self, connectionstring="mongodb://admincancer:2563crRT8@localhost:27017/"):
        #self.connectionstring = connectionstring
        self.conn = MongoClient(connectionstring)
        self.db = self.conn.CancerDB

        self.collection_cancers = self.db.cancers
        self.collection_patients = self.db.patients
        
    
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

    #def add_patient(self, name, email):

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
        item_count = self.collection_patients.count_documents({'name': patient.name})
        
        if item_count == 0:
            json_patient = self._get_json_patient(patient)
            return self.collection_patients.insert_one(json_patient).inserted_id
        
        return self.get_patient_by_name(patient.name)

    def get_patient_by_id(self, patient_id):
        result = list(self.collection_patients.find({'_id':ObjectId(patient_id)}))
        if len(result) == 0:
            return None
        return self._get_patient(result[0])
    
    def get_patient_by_name(self, name):
        return self.collection_patients.find({'name':name})

    def update_patient(self,patient):
        self.collection_patients.update_one({"_id": patient.id}, 
                                               {"$set":
                                                       {"image": patient.image,
                                                       "is_diagnosed": patient.is_diagnosed,
                                                       "has_cancer": patient.has_cancer}})

    def insert_first_data(self):
        post_data = {
            'name': 'Martin Smith',
            'image': "https://www.bootstrapdash.com/demo/breeze/angular/preview/demo_1/assets/images/faces/face1.jpg",
            'is_diagnosed': True,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Jimmy Nelson',
            'image': "https://www.bootstrapdash.com/demo/breeze/angular/preview/demo_1/assets/images/faces/face13.jpg",
            'is_diagnosed': True,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Carrie Parker',
            'image': "https://www.bootstrapdash.com/demo/breeze/angular/preview/demo_1/assets/images/faces/face11.jpg",
            'is_diagnosed': False,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Harry Holloway',
            'image': "https://www.bootstrapdash.com/demo/breeze/angular/preview/demo_1/assets/images/faces/face7.jpg",
            'is_diagnosed': True,
            'has_cancer':True
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Ethel Doyle',
            'image': "https://www.bootstrapdash.com/demo/futureui/template/images/faces/face1.jpg",
            'is_diagnosed': True,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Cameron',
            'image': "https://www.bootstrapdash.com/demo/futureui/template/images/faces/face3.jpg",
            'is_diagnosed': True,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Jose Ball',
            'image': "https://www.bootstrapdash.com/demo/futureui/template/images/faces/face4.jpg",
            'is_diagnosed': True,
            'has_cancer':True
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Jared Carr',
            'image': "https://www.bootstrapdash.com/demo/futureui/template/images/faces/face6.jpg",
            'is_diagnosed': False,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'David Grey',
            'image': "https://www.bootstrapdash.com/demo/purple/angular/preview/demo_1/assets/images/faces/face1.jpg",
            'is_diagnosed': False,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Stella Johnson',
            'image': "https://www.bootstrapdash.com/demo/purple/angular/preview/demo_1/assets/images/faces/face2.jpg",
            'is_diagnosed': True,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Marina Michel',
            'image': "https://www.bootstrapdash.com/demo/purple/angular/preview/demo_1/assets/images/faces/face3.jpg",
            'is_diagnosed': True,
            'has_cancer':True
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'John Doe',
            'image': "https://www.bootstrapdash.com/demo/purple/angular/preview/demo_1/assets/images/faces/face4.jpg",
            'is_diagnosed': True,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Connor Chandler',
            'image': "https://www.bootstrapdash.com/demo/celestial/template/images/faces/face31.png",
            'is_diagnosed': False,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Russell Floyd',
            'image': "https://www.bootstrapdash.com/demo/celestial/template/images/faces/face32.png",
            'is_diagnosed': True,
            'has_cancer':False
        }
        self.insert_patient(post_data)

        post_data = {
            'name': 'Allen Moreno',
            'image': "https://www.bootstrapdash.com/demo/star-admin-pro/src/assets/images/faces/face1.jpg",
            'is_diagnosed': False,
            'has_cancer':False
        }
        result = self.insert_patient(post_data)

        post_data = {
            'name': 'Fukuyo Kazutoshi',
            'image': "https://www.bootstrapdash.com/demo/star-admin-pro/src/assets/images/faces/face4.jpg",
            'is_diagnosed': True,
            'has_cancer':False
        }
        result = self.insert_patient(post_data)

        return result

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

#if __name__ == "__main__":
#     api = CancerDBAPI()
#     result = api.insert_first_data()
#     if result != None:
#         print("result : ", result.inserted_id)