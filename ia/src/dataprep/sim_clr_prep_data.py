
import tensorflow as tf
import numpy as np
import os
import cv2
import glob 
class TFRecordHandler:
    def write(self, input_path, output_path, extension='jpg'):
        if not input_path:
            raise Exception("input must be provided")
        if not os.path.isdir(input_path):
            raise Exception("input_path doesn't exist")
        if not output_path:
            raise Exception("output_path must be provided")
        if not output_path.endswith('.tfrecords'):
            raise Exception("output_path must be of extension .tfrecords")

        paths = glob.glob(input_path + '\*.' + extension, recursive = True)

        counter = 0
        with tf.io.TFRecordWriter(output_path) as writer:
                for index, value in enumerate(paths):
                    img = self.load_image(paths[index])
                    image_h = img.shape[0]
                    image_w = img.shape[1]
                    if (image_h != 50) | (image_w != 50):
                        continue
                    counter += 1
                    example = tf.train.Example(
                    features=tf.train.Features(
                        #feature = get_feature(img)
                        feature = {
                            'height': self._int64_feature(image_h),
                            'width': self._int64_feature(image_w),
                            'image_raw': self._bytes_feature(img.tostring())}
                    ))
                    writer.write(example.SerializeToString())
                    print('\r{:.1%}'.format((index+1)/len(paths)), end='')
        print("\ncounter :", counter)

    def load_image(self, path):
        img = cv2.imread(path)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img.astype(np.float32)
        img = img.astype(np.uint8)
        #np.uint8
        return img
    
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _floats_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
