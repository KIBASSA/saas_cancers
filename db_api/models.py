
import json
from json import JSONEncoder
from bson.objectid import ObjectId
import datetime
class Patient:
    def __init__(self, id,name,image = "",is_diagnosed=False,has_cancer=False,registration_date=False,diagnosis_date=False, cancer_images=[]):
        self.id = id
        self.name = name
        self.image = image
        self.is_diagnosed = is_diagnosed
        self.has_cancer = has_cancer
        self.registration_date = registration_date
        self.diagnosis_date = diagnosis_date
        self.cancer_images = cancer_images

class PatientEncoder(JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, datetime.date):
            return dict(year=o.year, month=o.month, day=o.day)
        elif isinstance(o, ObjectId):
            return str(o)
        else:
            return o.__dict__