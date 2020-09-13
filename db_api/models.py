
import json
from json import JSONEncoder
import datetime
class Patient:
    def __init__(self, id,name,image,is_diagnosed,has_cancer,registration_date,diagnosis_date):
        self.id = id
        self.name = name
        self.image = image
        self.is_diagnosed = is_diagnosed
        self.has_cancer = has_cancer
        self.registration_date = registration_date
        self.diagnosis_date = diagnosis_date

class PatientEncoder(JSONEncoder):
        def default(self, o):
            if isinstance(o, datetime.date):
                return dict(year=o.year, month=o.month, day=o.day)
            else:
                return o.__dict__