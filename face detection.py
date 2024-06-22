import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing import image

# Data augmentation and rescaling for training set
train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)

# Directory path for training data
train_path = r" "
training_set = train_gen.flow_from_directory(
    train_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Data rescaling for test set
test_gen = ImageDataGenerator(rescale=1./255)
test_path = r" "
test_set = test_gen.flow_from_directory(
    test_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Load the VGG16 model with pre-trained weights, excluding the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the layers in the base model to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_set, validation_data=test_set, epochs=25)

test_image_path = r" "
test_image = image.load_img(test_image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Rescale the image to match training set preprocessing

# Predict the class of the test image
result = model.predict(test_image)

# Print the class indices
print(training_set.class_indices)

# Determine and print the prediction based on the result
if result[0][0] > 0.5:
    prediction = 'real'
else:
    prediction = 'fake'
print(prediction)