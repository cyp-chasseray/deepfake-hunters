############### IMPORTS ##############

############## ENCODED DATA ############

def rescale_data(X):
    X = X.astype('float32')
    X /= 255
    return X