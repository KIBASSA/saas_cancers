from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
import ntpath
import inspect
import os
class BlobStorageManager:
    def __init__(self, connection_string="DefaultEndpointsProtocol=https;AccountName=diagnozstorage;AccountKey=SWWLDWxC6xjhWuNTblGdkOT6jAPcpA0W1LzowyginzEsibTHqla2xurPgWeRtcCzO2Rb0KXpTn3KXdn38EYTag==;EndpointSuffix=core.windows.net"):
        self.connection_string = connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
    
    def upload(self, blob_container, file_path, overwrite = False):

        file_name = ntpath.basename(file_path)
        print("file_name :", file_name)
        blob_client = self.blob_service_client.get_blob_client(container=blob_container, blob=file_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)
    
    def props(self, cls):
        """
        allows to retrieve the properties and attributes of an object
        """
        return [i for i in cls.__dict__.keys() if i[:1] != '_']

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
    
    def delete(self,blob_container,  file_path):
        """deletes the file from the blob storage

        Arguments:
            blob_container {str} -- container (or folder) in blob storage 
            file_path {str} -- [description]
        """

        if os.path.isabs(file_path):
            file_name = ntpath.basename(file_path)
        else:
            file_name = file_path
        # Create a blob client using the local file name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=blob_container, blob=file_name)

        print("\Deleting from Azure Storage:\n\t" + file_name)

        # Upload the created file
        blob_client.delete_blob()

#if __name__ == "__main__":
#    blob_service_client = BlobServiceClient.from_connection_string("connection_string")
#    container_client = blob_service_client.get_container_client("blob_container")
#    result = list(container_client.list_blobs(prefix="prefix"))
#    for blob in container_client.list_blobs(prefix="prefix"):
#        files.append(blob)
    #print(ItemPaged)
#     app = BlobStorageManager()
#     #app.get_list_file("diagnoz/patients/5f76ae1ab18cb6ab2172b606/cancer/breast/images")
#     print(app.get_list_file("diagnoz", "patients/5f76ae1ab18cb6ab2172b606/cancer/breast/images"))