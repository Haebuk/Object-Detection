import tensorflow as tf
from mlp_mixer_model import MLPMixer
from argparse import ArgumentParser
import os
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    home_dir = os.getcwd()
    parser.add_argument('--test-file-path', default=f"{home_dir}/data/test", type=str, required=True)
    parser.add_argument('model-folder', default=f"{home_dir}/model/mlp", type=str)
    parser.add_argument('--image-size', default=150, type=int)

    args = parser.parse_args()
    print('='*20)
    print('Predict using MLP Mixer for image path: {}'.format(args.test_file_path))
    print('='*20)

    mlpmixer = tf.keras.model.load_model(args.model_folder)

    image = tf.keras.preprocessing.image.load_img(args.test_file_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # convert single image to a batch
    x = tf.image.resize(
        input_arr, [args.image_size, args.image_size]
    )

    predictions = mlpmixer.predict(x)

    print('='*10 + 'Prediction Result' + '='*10)
    print('Output Softmax: {}'.format(predictions))
    print('This image belongs to class: {}'.format(np.argmax(predictions), axis=1))