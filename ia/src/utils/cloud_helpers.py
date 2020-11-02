
from azure.storage.blob import BlobServiceClient
import ntpath
class BlobStorageHandler(object):
    def __init__(self, connection_string="DefaultEndpointsProtocol=https;AccountName=diagnozstorage;AccountKey=SWWLDWxC6xjhWuNTblGdkOT6jAPcpA0W1LzowyginzEsibTHqla2xurPgWeRtcCzO2Rb0KXpTn3KXdn38EYTag==;EndpointSuffix=core.windows.net"):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    def upload(self, blob_container, file_path, overwrite = False):
        file_name = ntpath.basename(file_path)
        print("file_name :",file_name)
        blob_client=self.blob_service_client.get_blob_client(container=blob_container,blob=file_name)
        with open(file_path,"rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)