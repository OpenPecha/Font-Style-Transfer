#initialize
import tensorflow as tf
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
handwritten_dir = '/chinese-handwriting/CASIA-HWDB_Train/Train'
computer_font_dir = '/Generated_Characters'
handwritten_images = []
for root, dirs, files in os.walk(handwritten_dir):
    for dir in dirs:
        character = dir  # Each subfolder name is the character
        character_dir = os.path.join(root, dir)
        image_files = glob.glob(os.path.join(character_dir, '*.png'))
        for image_file in image_files:
            handwritten_images.append((character, image_file))
computer_font_images = []
for filename in os.listdir(computer_font_dir):
    if filename.endswith('.png'):
        character = os.path.splitext(filename)[0]  # Extract character from filename
        image_file = os.path.join(computer_font_dir, filename)
        computer_font_images.append((character, image_file))
paired_images = []
for handwritten_char, handwritten_image_path in handwritten_images:
    for computer_font_char, computer_font_image_path in computer_font_images:
        if handwritten_char == computer_font_char:
            paired_images.append((handwritten_image_path, computer_font_image_path))
            break  
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  
    image = tf.image.resize(image, (58, 58))
    image = tf.cast(image, tf.float32) / 255.0 
    return image
dataset = tf.data.Dataset.from_tensor_slices(paired_images)
dataset = dataset.map(lambda x: (load_image(x[0]), load_image(x[1])))
SHUFFLE_BUFFER_SIZE = len(dataset)
dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
dataset = dataset.cache()
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
BATCH_SIZE = 1024
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#define GAN
IMAGE_SIZE = (58, 58, 1)
def build_generator():
    model = models.Sequential([
        layers.Input(shape=IMAGE_SIZE),
        layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', activation='sigmoid')  # Output layer (1 channel, grayscale)
    ], name='generator')
    return model
def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=IMAGE_SIZE),
        layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')  # Output layer (binary classification: real or fake)
    ], name='discriminator')
    return model
discriminator = build_discriminator()
discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
generator = build_generator()
computer_font_image_input = layers.Input(shape=IMAGE_SIZE)
generated_handwritten_image = generator(computer_font_image_input)
discriminator.trainable = False  
gan_output = discriminator(generated_handwritten_image)
gan = models.Model(computer_font_image_input, gan_output, name='gan')
gan.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
def generate_samples(generator, input_image):
    generated_images = generator.predict(input_image[np.newaxis, ...])
    return generated_images

import logging
from tqdm import tqdm
import numpy as np

# Configure logging to file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='gan_training.log',  # Specify log file
                    filemode='w')  # 'w' mode overwrites existing log file

# Define constants
EPOCHS = 1000
SAVE_EVERY = 10
for epoch in tqdm(range(EPOCHS), desc='Training', unit='epoch'):
    for batch in dataset:
        computer_font_images, handwritten_images = batch
        real_labels = np.ones((BATCH_SIZE, 1))
        d_loss_real = discriminator.train_on_batch(handwritten_images, real_labels)
        fake_labels = np.zeros((BATCH_SIZE, 1))
        generated_images = generator.predict(computer_font_images)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        g_loss = gan.train_on_batch(computer_font_images, real_labels)
    logging.info(f'Epoch {epoch + 1}/{EPOCHS} - Discriminator Loss: Real {d_loss_real}, Fake {d_loss_fake}')
    logging.info(f'Epoch {epoch + 1}/{EPOCHS} - Generator Loss: {g_loss}')
    if (epoch + 1) % SAVE_EVERY == 0:
        logging.info(f'Saving model checkpoint at epoch {epoch + 1}')
        generator.save(f'generator_epoch_{epoch + 1}.h5')
        discriminator.save(f'discriminator_epoch_{epoch + 1}.h5')
logging.info('GAN training completed.')