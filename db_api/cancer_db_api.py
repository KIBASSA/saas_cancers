from pymongo import MongoClient
from pymongo import errors
import pymongo 
from datetime import date

class CancerDBAPI:
    def __init__(self, connectionstring="='mongodb://admincancer:2563crRT8@localhost:27017/'"):
        self.conn = connectionstring
        self.conn = MongoClient(connectionstring)
        try :
            self.db = self.conn['CancerDB']
        except :
            self.db = self.conn.CancerDB
        self.collection_cancers = self.db['cancers']
    
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