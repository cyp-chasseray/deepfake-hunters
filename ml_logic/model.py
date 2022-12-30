######################### Imports ##########################
#Imports pour le model
from tensorflow.keras import Sequential, layers
from keras.models import Model
# from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

#Imports pour les graphiques
import matplotlib.pyplot as plt
import pathlib
# %matplotlib inline


######################## Model CNN ############################
#model initialization
def initialize_model():
    model = Sequential()

    model.add(Rescaling(1./255, input_shape=(224,224,3)))
    model.add(layers.Conv2D(8, (3,3), activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(8, (5,5), activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(16, (5,5), activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D(pool_size=(4,4)))
    model.add(layers.Conv2D(64, (2,2), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    ### Model compilation
    model.compile(optimizer = "adam",
                  loss = "binary_crossentropy",
                  metrics="accuracy")
    return model

#Model Training
def train_model(model: Model, X_train,y_train, X_test, y_test, X_val, y_val):
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    #Early stopping
    es = EarlyStopping(patience = 15, restore_best_weights = True)

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=500,
                        batch_size=32,
                        callbacks=[es],
                        verbose=1)

    print (model.evaluate(X_test, y_test))
    return model, history

def save_model(model):
    model.save("./local_model")
    print("model saved")

###########################"" Visualisation graphique ########################

# #Recupération history
# history.__dict__

#Plot Loss et Accuracy
def plot_loss_accuracy(history, title=None):
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    # --- LOSS ---

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    #ax[0].set_ylim((0,3))
    ax[0].legend(['Train', 'Test'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- ACCURACY

    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])
    ax[1].set_title('Model Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='best')
    ax[1].set_ylim((0,1))
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    if title:
        fig.suptitle(title)





#ANOTHER MODEL WITH SIMILAR PERFORMANCES

# def initialize_model():

#     model = Sequential()
#     model.add(Rescaling(1./255, input_shape=(224,224,3)))

# #     model.add(layers.RandomRotation(0.2))

#     model.add(layers.Conv2D(8, (3,3), activation="relu"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPool2D(pool_size=(2,2)))


#     model.add(layers.Conv2D(8, (5,5), activation="relu"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPool2D(pool_size=(2,2)))


#     model.add(layers.Conv2D(16, (5,5), activation="relu"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPool2D(pool_size=(4,4)))


#     model.add(layers.Conv2D(64, (2,2), activation="relu"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPool2D(pool_size=(2,2)))


#     model.add(layers.Flatten())

#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(16, activation='relu'))

#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(1, activation='sigmoid'))

#     ### Model compilation
#     model.compile(optimizer = "adam",
#                   loss = "binary_crossentropy",
#                   metrics="accuracy")
#     return model
