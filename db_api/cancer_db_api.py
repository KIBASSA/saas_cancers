from pymongo import MongoClient
from pymongo import errors
import pymongo 
from datetime import date
from models import Patient

class CancerDBAPI:
    def __init__(self, connectionstring="mongodb://admincancer:2563crRT8@localhost:27017/"):
        self.conn = connectionstring
        self.conn = MongoClient(connectionstring)
        try :
            self.db = self.conn['CancerDB']
        except :
            self.db = self.conn.CancerDB
        self.collection_cancers = self.db['cancers']
        self.collection_patients = self.db['patients']
        print("KEBAAAA")
    
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
            
            #result = self.collection_patients.find({"is_diagnosed": False})
            result = list(self.collection_patients.find({}))
            print("*************result :", len(result))
            for item in result:
                print("KKKKKKKKKKKKKKKK")
                patient = Patient(id=item["_id"],name=item["name"],image=item["image"],is_diagnosed=item["is_diagnosed"], has_cancer=item["has_cancer"])
                patients.append(patent)
        except Exception as e:
            raise Exception(e)
        
        return patients