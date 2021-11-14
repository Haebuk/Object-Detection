import os
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from mlp_mixer_model import MLPMixer
import logging
logging.basicConfig(level=logging.DEBUG)


# python /drone/mlp-mixer/train.py
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()
    parent_dir = os.path.dirname(home_dir)
    print(parent_dir)
    parser.add_argument("--train-folder", default='{}/dataset/intel/seg_train/'.format(home_dir), type=str)
    parser.add_argument("--valid-folder", default='{}/dataset/intel/seg_test/'.format(home_dir), type=str)
    parser.add_argument("--model-folder", default='{}/model/mlp/'.format(home_dir), type=str)
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=3000, type=int)
    parser.add_argument("--dc", default=2048, type=int, help='Token-mixing units')
    parser.add_argument("--ds", default=256, type=int, help='Channel-mixing units')
    parser.add_argument("--c", default=512, type=int, help='Projection units')
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--patch-size", default=32, type=int)
    parser.add_argument("--num-of-mlp-blocks", default=8, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--validation-split", default=0.2, type=float)
    parser.add_argument("--image-channels", default=3, type=int)

    args = parser.parse_args()

    print('='*20)
    for i, arg in enumerate(vars(args)):
        print(f"{i}.{arg}: {vars(args)[arg]}")
    print('='*20)

    train_folder = args.train_folder
    valid_folder = args.valid_folder

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_folder,
        subset='training',
        seed=42,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_folder,
        subset='validation',
        seed=42,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
    )

    assert args.image_size * args.image_size % (args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'

    S = (args.image_size * args.image_size) // (args.patch_size * args.patch_size)

    mlpmixer = MLPMixer(
        args.patch_size,
        S,
        args.c,
        args.ds,
        args.dc,
        args.num_of_mlp_blocks,
        args.image_size,
        args.batch_size,
        args.num_classes,
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # earlystopping = EarlyStopping(monitor='val_loss', patience=20)
    lr_scheduler = tf.keras.experimental.CosineDecayRestarts(
        args.learning_rate, 
        first_decay_steps=100, 
        t_mul=2.0, 
        m_mul=1.0, 
        alpha=0.0,
        name=None
)

    adam = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    mlpmixer.compile(
        optimizer=adam, loss=loss_object, metrics=['accuracy']
    )

    # mlpmixer.fit(
    #     x_train, y_train,
    #     epochs=args.epochs,
    #     validation_data=(x_val, y_val),
    # )

    mlpmixer.fit(
        train_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=val_ds,
        # callbacks=[earlystopping]
        )

    mlpmixer.save(args.model_folder)