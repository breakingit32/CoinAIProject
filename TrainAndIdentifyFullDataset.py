import os
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# reduce system log presentation in console.
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Starting File location - All files - Organized in folders NAMED with Classifications.
data_dir = "F:/PROJECTS/CentSearch/data_start_full/"
data_dir = pathlib.Path(data_dir)
print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

img_height = 100
img_width = 100
batch_size = 2

# ==================================================== #
#             Using dataset_from_directory             #
# ==================================================== #
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",  # int, categorical, binary
    # class_names=['obverse_linc_142', 'obverse_linc_16', . . . etc]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",  # int, categorical, binary
    # class_names=['obverse_linc_142', 'obverse_linc_16', . . . etc]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

# Array containing all the FOLDER names - Index is output of Network
class_names = ds_train.class_names
print(class_names)

# Number fo Classifications is the number of elements in the above array.
num_classes = len(class_names)


# Vary the brightness of the images to create more training data elements.
def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


ds_train = ds_train.map(augment)

# Input 100 px square GRAYSCALE image - Output index location of CLASS_NAMES array.
model = keras.Sequential(
    [
        layers.Input((100, 100, 1)),
        layers.Conv2D(128, 3, padding="same"),
        layers.Conv2D(64, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.Conv2D(16, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(360),
    ]
)


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

# Load model from previous epochs
model = keras.models.load_model('cent_obverse_model_full/')

# Output model structure summary to console
model.summary()

# Output status bar during epochs
AUTOTUNE = tf.data.AUTOTUNE

train_ds = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = ds_validation.cache().prefetch(buffer_size=AUTOTUNE)

# Train and write model data after each epoch
model.fit(
    ds_train,
    validation_data=ds_validation,
    epochs=12,
    callbacks=[keras.callbacks.ModelCheckpoint("cent_obverse_model_full")]
)

# Save model again after all epochs have run
model.save('cent_obverse_model_full/')

# Test image to check against the model.
coin_image_path = 'images/051_obverse_lincoln_rotate_204.jpg'

# Open image as GRAYSCALE - Default is RGB
img = tf.keras.utils.load_img(
    coin_image_path, color_mode='grayscale', target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Calculate prediction from image provided
predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

# Output the results
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)