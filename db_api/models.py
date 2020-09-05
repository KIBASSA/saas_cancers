
import json
from json import JSONEncoder

class Patient:
    def __init__(self, id,name,image,is_diagnosed,has_cancer):
        self.id = id
        self.name = name
        self.image = image
        self.is_diagnosed = is_diagnosed
        self.has_cancer = has_cancer

class PatientEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__