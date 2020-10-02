from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import ntpath
class BlobStorageManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
    
    def upload(self, blob_container, file_path, overwrite_v = False):

        file_name = ntpath.basename(file_path)
        print("file_name :", file_name)
        blob_client = self.blob_service_client.get_blob_client(container=blob_container, blob=file_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite_v)

