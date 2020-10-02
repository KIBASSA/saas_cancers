import tempfile
import os
from mimetypes import guess_extension, guess_type
import base64
class Consts:
    BREAST = "breast"
    DIAGNOZ_PATIENTS = "diagnoz/patients"

class PatientImageURLBuilder:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.host = "https://diagnozstorage.blob.core.windows.net/"

    def get_image(self,patient,img_data):
        ext = guess_extension(guess_type(img_data)[0])
        return "{0}/{1}/{2}/profile".format(self.host, Consts.DIAGNOZ_PATIENTS, patient.id)

class PatientImageCloudManager:
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager
        self.host = "https://diagnozstorage.blob.core.windows.net"
        
    def _get_container(self,patient,img_data):
        return "{0}/{1}/profile".format(Consts.DIAGNOZ_PATIENTS, patient.id)

    def _get_uploaded_image_file(self,blob_container, file_name):
        return "{0}/{1}/{2}".format(self.host, blob_container, file_name)

    def _clean_image(self, image):
        exts = ["jpeg", "jpg", "png", "gif", "tiff"]
        for ext in exts:
            image = image.replace("data:image/{0};base64,".format(ext), "")
        return image 

    def upload_profile(self, patient, img_data):
        uploaded_image = None
        ext = guess_extension(guess_type(img_data)[0])
        with tempfile.TemporaryDirectory() as dir:
            local_file_name = "{0}{1}".format(patient.id, ext)
            local_full_path_file = os.path.join(dir, "{0}{1}".format(patient.id, ext))
            img_data = self._clean_image(img_data)
            with open(local_full_path_file, "wb") as fh:
                fh.write(base64.decodestring(img_data.encode()))
            blob_container = self._get_container(patient, img_data)
            self.blob_manager.upload(blob_container, local_full_path_file, overwrite_v = True)
            uploaded_image = "{0}/{1}/{2}".format(self.host, blob_container, local_file_name)
            print("uploaded_image :", uploaded_image)
        return uploaded_image



   