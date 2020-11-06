
from PIL import Image, ImageOps
from io import BytesIO
import base64
import numpy as np

class Predictor(object):
    def predict(self, model, images_base64, image_shape, labels):
        responses = []
        for img_file_name, base64Img in images_base64.items():
            decoded_img = base64.b64decode(image_base64)
            img_buffer = BytesIO(decoded_img)
            imageData = Image.open(img_buffer).convert("RGB")
            
            img = ImageOps.fit(imageData, image_shape, Image.ANTIALIAS)
            img_conv = np.array(img)

            x_test = img_conv / 255.0
            x_test = np.expand_dims(x_test, 0)

            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=-1)

            resp = {img_file_name: str(labels[y_pred[0]])}
            responses.append(resp)
        return json.dumps(responses)
