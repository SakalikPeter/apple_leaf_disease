import tensorflow as tf
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

def build_model(input_shape, n_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    return model

def train_model(model: models.Sequential, compile_params: dict, fit_params: dict, train_generator, validation_generator, epochs):
    model.compile(**compile_params)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        **fit_params
    )

    return history

def plot_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label='Training Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label='Training Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('tr.png')

def predict(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

def visualize_predictions(model, test_generator, class_names):
    plt.figure(figsize=(15, 15))
    for images, labels in test_generator:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])

            predicted_class, confidence = predict(model, images[i], class_names)
            actual_class = class_names[int(labels[i])]


            plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")

            plt.axis("off")
        break
    plt.savefig('pred.png')

def save_model(model: models.Sequential, filepath):
    model.save(filepath)

def create_data_generator(dir: str, generator_params: dict, datagen_params: dict):
    datagen = ImageDataGenerator(**generator_params)

    generator = datagen.flow_from_directory(
        directory=dir,
        **datagen_params
    )

    return generator

def load_parameters(file_path):
    with open(file_path, 'r') as f:
        parameters = json.load(f)
    return parameters

def evaluate_model(model, test_generator):
    return model.evaluate(test_generator)

def get_loss_function(loss_name):
    if loss_name == "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def main():

    # Load parameters from JSON file
    parameters = load_parameters('deep_learning/configs/train_config.json')

    train_attr = parameters['base']
    datagen_attrs = parameters['datagen']
    generator_attrs = parameters['generator']
    dataset_paths = parameters['dataset']
    compile_attrs = parameters['compile']
    fit_attrs = parameters['fit']
    save_attrs = parameters['save']

    epochs = train_attr['epochs']
    image_size = train_attr['image_size']
    channels = train_attr['channels']
    n_classes = train_attr['n_classes']

    datagen_attrs['target_size'] = (image_size, image_size)
    generator_attrs['rescale'] = 1./255
    compile_attrs['loss'] = get_loss_function(compile_attrs['loss'])
    input_shape = (image_size, image_size, channels)

    # Image Generators
    train_generator = create_data_generator(
        dataset_paths['train'], generator_attrs, datagen_attrs
    )
    validation_generator = create_data_generator(
        dataset_paths['val'], generator_attrs, datagen_attrs
    )
    test_generator = create_data_generator(
        dataset_paths['test'], generator_attrs, datagen_attrs
    )

    # Model Builder
    model = build_model(input_shape, n_classes)

    # Train Builded Model
    history = train_model(model, compile_attrs, fit_attrs, train_generator, validation_generator, epochs)

    # Evaluate Model
    scores = evaluate_model(model, test_generator)

    # Save Model
    save_model(model, save_attrs['file'].format(datetime.now()))
    
    # print("loss, acc score: ", scores)

    # print("params: ", history.params)
    # print("keys: ", history.history.keys())

    # plot_history(history, EPOCHS)

    # visualize_predictions(model, test_generator, class_names)

    

if __name__ == "__main__":
    main()