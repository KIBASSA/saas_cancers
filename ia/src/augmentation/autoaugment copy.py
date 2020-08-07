import inspect
import math
import tensorflow.compat.v1 as tf
from tensorflow.contrib import image as contrib_image
from tensorflow.contrib import training as contrib_training
def distort_image_with_randaugment(image, num_layers, magnitude):
  """Applies the RandAugment policy to `image`.
  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].
  Returns:
    The augmented version of `image`.
  """
  replace_value = [128] * 3
  tf.logging.info('Using RandAug.')
  augmentation_hparams = contrib_training.HParams(
      cutout_const=40, translate_const=100)
  available_ops = [
      'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
      'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
      'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']

  for layer_num in range(num_layers):
    op_to_select = tf.random_uniform(
        [], maxval=len(available_ops), dtype=tf.int32)
    random_magnitude = float(magnitude)
    with tf.name_scope('randaug_layer_{}'.format(layer_num)):
      for (i, op_name) in enumerate(available_ops):
        prob = tf.random_uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
        func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                           replace_value, augmentation_hparams)
        image = tf.cond(
            tf.equal(i, op_to_select),
            # pylint:disable=g-long-lambda
            lambda selected_func=func, selected_args=args: selected_func(
                image, *selected_args),
            # pylint:enable=g-long-lambda
            lambda: image)
  return image