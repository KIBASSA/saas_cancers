from pymongo import MongoClient
from pymongo import errors
import pymongo 
from datetime import date
from models import Patient

class CancerDBAPI:
    def __init__(self, connectionstring="mongodb://admincancer:2563crRT8@localhost:27017/"):
        #self.connectionstring = connectionstring
        self.conn = MongoClient(connectionstring)
        self.db = self.conn.CancerDB

        self.collection_cancers = self.db.cancers
        self.collection_patients = self.db.patients

        #bulk insert
        #print("bulk insert")
        #self.insert_first_data()
        
    
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

    def get_undiagnosed_patients(self):
        patients = []
        try:
            result = self.collection_patients.find({"is_diagnosed":False})
            for item in result:
                patient = Patient(id=str(item["_id"]),
                                    name=item["name"],
                                        image=item["image"],
                                            is_diagnosed=item["is_diagnosed"], 
                                                has_cancer=item["has_cancer"],
                                                  registration_date=item["registration_date"],
                                                   diagnosis_date=item["diagnosis_date"])
                
                patients.append(patient)
        except Exception as e:
            raise Exception(e)
        
        return patients
    
    def insert_patient(self, patient):
        item_count = self.collection_patients.count_documents({'name': patient['name']})
        if item_count == 0:
            result = self.collection_patients.insert_one(patient)

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
