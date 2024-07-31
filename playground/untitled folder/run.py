import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from PIL import Image
import numpy as np
import datasets
from tqdm import tqdm
import pickle
from datasets import Dataset
from datasets import load_dataset
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import logging
from tensorflow.keras.models import load_model

#initialization
image_folder = "/local_dir/Train_Images"
hf_dataset_name = "ta4tsering/Lhasa_kanjur_transcription_datasets"
hf_dataset = datasets.load_dataset(hf_dataset_name, split="train")
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to black and white
    image = image.resize((1400, 70))
    image = np.array(image) / 255.0  # Normalize
    return image
def transcription_to_vector(transcription):
    tokens = tokenizer(transcription, return_tensors="tf", padding="max_length", max_length=512, truncation=True)
    inputs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }
    outputs = model(**inputs)
    vector = outputs.pooler_output  # Extract the pooled output as the vector representation
    return vector
tokenizer = AutoTokenizer.from_pretrained("openpecha/tibetan_RoBERTa_S_e6")
model = TFAutoModel.from_pretrained("openpecha/tibetan_RoBERTa_S_e6")

#dataset multi-download
logging.basicConfig(filename='/download-training2.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
def convert_to_tensors(item):
    image_tensor = tf.convert_to_tensor(item['image'])
    vector_tensor = tf.convert_to_tensor(item['vector'])
    return image_tensor, vector_tensor
tf_dataset = None
for dataset_number in range(1, 14):
    dataset_name = f"norbujam/LG-dataset-{dataset_number}"
    dataset = load_dataset(dataset_name)['train']
    image_tensors = []
    vector_tensors = []
    total_items = len(dataset)
    pbar = tqdm(total=total_items, desc=f"Downloading and converting {dataset_name}", unit="item")
    update_counter = 0
    for idx, item in enumerate(dataset):
        image_tensor, vector_tensor = convert_to_tensors(item)
        image_tensors.append(image_tensor)
        vector_tensors.append(vector_tensor)
        pbar.update(1)
        update_counter += 1
        if update_counter == 1000:
            logging.info(f'Converted {len(image_tensors)} items from {dataset_name} to TensorFlow tensors')
            update_counter = 0 
    pbar.close()
    current_tf_dataset = tf.data.Dataset.from_tensor_slices((image_tensors, vector_tensors))
    if tf_dataset is None:
        tf_dataset = current_tf_dataset
    else:
        tf_dataset = tf_dataset.concatenate(current_tf_dataset)
    logging.info(f'Converted {dataset_number} repo')

#define cgan
def build_generator(vector_dim, noise_dim, img_shape):
    input_vector = layers.Input(shape=(vector_dim,))
    input_noise = layers.Input(shape=(noise_dim,))
    x = layers.Concatenate()([input_vector, input_noise])
    x = layers.Dense(35*700, activation='relu')(x)
    x = layers.Reshape((35,700,1))(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    output_img = layers.Conv2D(1, (5, 5), activation='tanh', padding='same')(x)
    return Model(inputs=[input_vector, input_noise], outputs=output_img)
def build_discriminator(vector_dim, img_shape):
    input_vector = layers.Input(shape=(vector_dim,))
    input_img = layers.Input(shape=img_shape)
    reshaped_img = layers.Reshape((img_shape[0], img_shape[1], 1))(input_img)
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='valid', activation='relu')(reshaped_img)
    x = layers.Conv2D(16, (5, 5), strides=(3, 3), padding='valid', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, input_vector])
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=[input_vector, input_img], outputs=output)
vector_dim = 768  
noise_dim = 100  
img_shape = (70, 1400)
generator = build_generator(vector_dim, noise_dim, img_shape)
discriminator = build_discriminator(vector_dim, img_shape)
generator = load_model('model_checkpoints/generator_epoch_100.h5')
discriminator = load_model('model_checkpoints/discriminator_epoch_100.h5')
cross_entropy = tf.keras.losses.BinaryCrossentropy()
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
@tf.function
def train_step(images, vectors):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([vectors, noise], training=True)
        real_output = discriminator([vectors, images], training=True)
        fake_output = discriminator([vectors, generated_images], training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss
def train(dataset, epochs, checkpoint_path, save_every=100):
    total_batches = len(dataset)
    for epoch in tqdm(range(epochs), desc='Training', unit='epoch'):
        epoch_gen_loss_avg = tf.metrics.Mean()
        epoch_disc_loss_avg = tf.metrics.Mean()
        for image_batch, vector_batch in tqdm(dataset):
            gen_loss, disc_loss = train_step(image_batch, vector_batch)
            epoch_gen_loss_avg.update_state(gen_loss)
            epoch_disc_loss_avg.update_state(disc_loss)
        logging.info(f'Epoch {epoch + 1}/{epochs} - Generator Loss: {epoch_gen_loss_avg.result()}, Discriminator Loss:{epoch_disc_loss_avg.result()}')
        if (epoch + 1) % save_every == 0:
            logging.info(f'Saving model checkpoint at epoch {epoch + 1}')
            generator.save(f'{checkpoint_path}/generator_epoch_{epoch + 1}.h5')
            discriminator.save(f'{checkpoint_path}/discriminator_epoch_{epoch + 1}.h5')
    logging.info(f'Training completed for {epochs} epochs')

#dataset shaping
tf_dataset = tf_dataset.cache()
tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
batch_size = 128
mydataset=tf_dataset.batch(batch_size, drop_remainder=True)
def reshape_image(image, label):
    label_reshaped = tf.squeeze(label, axis=1)
    return image, label_reshaped
mydataset = mydataset.map(reshape_image)
checkpoint_path = '/model_checkpoints2' 

train(mydataset, epochs=10000, checkpoint_path=checkpoint_path, save_every=10)