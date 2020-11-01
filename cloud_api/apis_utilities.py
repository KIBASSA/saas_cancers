from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
import ntpath
class BlobStorageManager:
    def __init__(self, connection_string="DefaultEndpointsProtocol=https;AccountName=diagnozstorage;AccountKey=SWWLDWxC6xjhWuNTblGdkOT6jAPcpA0W1LzowyginzEsibTHqla2xurPgWeRtcCzO2Rb0KXpTn3KXdn38EYTag==;EndpointSuffix=core.windows.net"):
        self.connection_string = connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
    
    def upload(self, blob_container, file_path, overwrite_v = False):

        file_name = ntpath.basename(file_path)
        print("file_name :", file_name)
        blob_client = self.blob_service_client.get_blob_client(container=blob_container, blob=file_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite_v)

    def get_list_file(self, blob_container, prefix):
        files = []
        container_client = self.blob_service_client.get_container_client(blob_container)
        try:
            for blob in container_client.list_blobs(prefix=prefix):
                if prefix in blob.name:
                    files.append(blob.name)
        except ResourceNotFoundError:
            #log...
            raise Exception("Container not found.")
        
        return files

#if __name__ == "__main__":
#     app = BlobStorageManager()
#     #app.get_list_file("diagnoz/patients/5f76ae1ab18cb6ab2172b606/cancer/breast/images")
#     print(app.get_list_file("diagnoz", "patients/5f76ae1ab18cb6ab2172b606/cancer/breast/images"))