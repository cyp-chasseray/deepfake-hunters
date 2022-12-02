from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.pyplot import imread
from tensorflow.image import resize
from tensorflow import expand_dims

# def load_data():
#     X_train = 0
#     y_train = 0
#     return X_train, y_train

# def initialize_model():
#     model = Sequential()
#     model.add(Rescaling(1./255, input_shape=(224,224,3)))
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

model = models.load_model('./model-abou')
# model.summary()
test_img = imread('./Images/fake2.jpg')
test_img_resized = resize(test_img, [224,224])
test_img_resized_scaled = test_img_resized/255.
image_final = expand_dims(test_img_resized_scaled, 0)
res = model.predict(image_final)
print(res)

# def fit_model(X_train, y_train):

#     es = EarlyStopping(patience=15)
#     history = model.fit(X_train, y_train,
#             batch_size = 32,
#             epochs = 500,
#             validation_data=(X_val, y_val),
#             callbacks = [es],
#             verbose = 0)
#     return history

# def predict(fitted_model, img_4d):
#     return fitted_model.predict(img_4d)
