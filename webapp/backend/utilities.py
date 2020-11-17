import tempfile
import os
from mimetypes import guess_extension, guess_type
import base64
import tempfile
import json
import requests
from datetime import datetime
import shutil
import traceback
import sys
class Consts:
    BREAST = "breast"
    DIAGNOZ = "diagnoz"
    DIAGNOZ_HUML = "diagnozhuml"
    DIAGNOZ_PATIENTS = "diagnoz/patients"

class PatientImageURLBuilder:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.host = "https://diagnozstorage.blob.core.windows.net/"

    def get_image(self,patient,img_data):
        ext = guess_extension(guess_type(img_data)[0])
        return "{0}/{1}/{2}/profile".format(self.host, Consts.DIAGNOZ_PATIENTS, patient.id)

class AnnotatedDataManager:
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager
        self.host = "https://diagnozstorage.blob.core.windows.net"
     
    def upload_data(self, annotated_data):
        with tempfile.TemporaryDirectory() as dir:
            annotated_data_file = os.path.join(dir,'annotated_data.json')
            with open(annotated_data_file, 'w') as outfile:
                json.dump(annotated_data, outfile)
            blob_container = "{0}/mldata/annotated_data/current".format(Consts.DIAGNOZ)
            self.blob_manager.upload(blob_container, annotated_data_file, overwrite = True)
            print("file {0} uploaded".format(annotated_data_file))

class SampedDataDataManager:
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager
        self.host = "https://diagnozstorage.blob.core.windows.net"
        self.sampled_data_file = "sampled_data.json"
        self.sampled_data_url = "{0}/diagnoz/mldata/sampled_data/current/{1}".format(self.host,self.sampled_data_file)
        self.ARCHIVE_SAMPLED_PATH = "{0}/mldata/sampled_data/archive".format(Consts.DIAGNOZ)
        self.CURRENT_SAMPLED_PATH = "{0}/mldata/sampled_data/current".format(Consts.DIAGNOZ)

    def get_sampled_data(self):
        data = []
        try:
            with tempfile.TemporaryDirectory() as dir:
                file_path = os.path.join(dir, self.sampled_data_file)
                r = requests.get(self.sampled_data_url, allow_redirects=True)
                open(file_path, "wb").write(r.content)
                with open(file_path) as f:
                    data = json.load(f)
        except Exception as e:
            print(e)
        
        return data
    
    def archive(self):
        try:
            with tempfile.TemporaryDirectory() as dir:
                sampled_data_path = os.path.join(dir, self.sampled_data_file)
                r = requests.get(self.sampled_data_url, allow_redirects=True)
                open(sampled_data_path, "wb").write(r.content)
                archive_name = "sampled_data_{0}{1}".format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"),".json")
                archived_file = os.path.join(dir, archive_name)
                #copy sampled file as archived file
                shutil.copyfile(sampled_data_path, archived_file)
                #upload sampled file to blob
                self.blob_manager.upload(self.ARCHIVE_SAMPLED_PATH, archived_file)
                #upload sampled file to blob
                self.blob_manager.delete(self.CURRENT_SAMPLED_PATH, sampled_data_path)
                #delete sampled file
                os.remove(archived_file)
        except Exception as e:
            print(traceback.format_exc())
            print(sys.exc_info()[2])


class PatientImageCloudManager:
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager
        self.host = "https://diagnozstorage.blob.core.windows.net"
        
    def _get_profile_container(self,patient,img_data):
        return "{0}/{1}/profile".format(Consts.DIAGNOZ_PATIENTS, patient.id)
    
    def _get_cancer_images_container(self, patient_id):
        return "{0}/{1}/cancer/breast/images".format(Consts.DIAGNOZ_PATIENTS, patient_id)

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
            blob_container = self._get_profile_container(patient, img_data)
            self.blob_manager.upload(blob_container, local_full_path_file, overwrite = True)
            uploaded_image = "{0}/{1}/{2}".format(self.host, blob_container, local_file_name)
            print("uploaded_image :", uploaded_image)
        return uploaded_image


    def upload_cancer_images(self, patient_id, db_images_count, images):
        uploaded_images = []
        index = db_images_count
        with tempfile.TemporaryDirectory() as dir:
            for img_data in images:
                print("img_data :", img_data)
                ext = guess_extension(guess_type(img_data)[0])
                local_file_name = "{0}_{1}_{2}".format(patient_id,index,ext)
                local_full_path_file = os.path.join(dir, local_file_name)
                img_data = self._clean_image(img_data)
                with open(local_full_path_file, "wb") as fh:
                    fh.write(base64.decodestring(img_data.encode()))
                blob_container = self._get_cancer_images_container(patient_id)
                self.blob_manager.upload(blob_container, local_full_path_file, overwrite = True)
                uploaded_image = "{0}/{1}/{2}".format(self.host, blob_container, local_file_name)
                uploaded_images.append(uploaded_image)
                print("uploaded_image :", uploaded_image)
                index = index + 1 
        return uploaded_images





   