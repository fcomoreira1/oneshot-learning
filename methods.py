import emnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

''' Data handling general functions '''

def seperate_fewshot(test_images, test_labels, n):
    oneshot_data = []
    classify_data = []
    for label in np.unique(test_labels):
        for num in np.random.choice(np.where(test_labels == label)[0], n, False):
            oneshot_data.append(num)
    temp = set(oneshot_data)
    for i in range(len(test_labels)):
        if not i in temp: classify_data.append(i)
    oneshot_images = test_images[oneshot_data]
    oneshot_labels = test_labels[oneshot_data]
    classify_images = test_images[classify_data]
    classify_labels = test_labels[classify_data]
    return oneshot_images, oneshot_labels, classify_images, classify_labels

''' Extended MNIST related functions '''

def show_emnist():
    # works on notebooks with matplotlib inline, at least
    images, labels = emnist.extract_training_samples('balanced')
    f, axes = plt.subplots(47, 5, figsize = (8, 30))
    for i in range(47):
        index = np.where(labels == i)[0][:5]
        for j in range(5):
            axes[i][j].imshow(images[index[j]])
            axes[i][j].axis('off')

def get_emnist(n, train, verbose, reshape = True):
    '''
    n:          The number of train labels used for background training.
                There are 47 labels, meaning 47-n are used for oneshot learning and testing.
    train:      The number of picture that would be feed to oneshot training.
                if is 1, oneshot training; if more, fewshot training and oppoturnities for knn with >1 parameter.
    verbose:    Wheather to print stuff out or not.
    '''
    images, labels = emnist.extract_training_samples('balanced')
    images = images.copy().astype('float')
    images /= 255
    if reshape: images = images.reshape(-1, 28*28)
    assert (1 <= n <= 45), 'Invalid choice of n'

    # divide train and test
    if verbose: print("======= Loading emnist data ... =======")
    seperate = np.unique(labels)
    np.random.shuffle(seperate)
    train_pick = np.where(np.isin(labels, seperate[:n]))[0]
    test_pick = np.where(np.isin(labels, seperate[n:]))[0]
    train_images = images[train_pick, :, :]
    train_labels = labels[train_pick]
    test_images = images[test_pick, :, :]
    test_labels = labels[test_pick]
    if verbose:
        print("Output shape: ", [i.shape for i in [train_images, train_labels, test_images, test_labels]])
        print("Train labels: ", np.unique(train_labels))
        print("Test labels: ", np.unique(test_labels))

    # further divide test into train oneshot and classify
    oneshot_images, oneshot_labels, classify_images, classify_labels = seperate_fewshot(test_images, test_labels, train)
    if verbose: print("======= Finished loading. =======")

    return train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels

''' Functions with linear methods '''

def test_PCA(train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels, 
             n, n_components = 32, verbose=False, train=1):
    
    if verbose: print("======= PCA method: Training and evaluating ... =======")
    if verbose: print("Learning background ...")
    pca = PCA(n_components=n_components)
    pca.fit(X=train_images)

    if verbose: print("Vectorizing ...")
    oneshot_images = pca.transform(oneshot_images)
    classify_images = pca.transform(classify_images)

    if verbose: print("Learning oneshot ...")
    nn = min(train, 5)
    neigh = KNeighborsClassifier(n_neighbors = nn)
    neigh.fit(oneshot_images, oneshot_labels)

    if verbose: print("Predicting ...")
    pred = neigh.predict(classify_images)

    if verbose:
        print("Accuracy: ", np.sum(pred == classify_labels)/len(classify_labels))
        print("======= PCA method: Finished =======")

    return np.sum(pred == classify_labels)/len(classify_labels)

def test_LDA(train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels, 
             n, n_components = 32, verbose=False, train=1):
    
    if verbose: print("======= LDA method: Training and evaluating ... =======")
    if verbose: print("Learning background ...")
    lda = LDA(n_components=n_components)
    lda.fit(X=train_images)

    if verbose: print("Vectorizing ...")
    oneshot_images = lda.transform(oneshot_images)
    classify_images = lda.transform(classify_images)

    if verbose: print("Learning oneshot ...")
    nn = min(train, 5)
    neigh = KNeighborsClassifier(n_neighbors = nn)
    neigh.fit(oneshot_images, oneshot_labels)

    if verbose: print("Predicting ...")
    pred = neigh.predict(classify_images)

    if verbose:
        print("Accuracy: ", np.sum(pred == classify_labels)/len(classify_labels))
        print("======= LDA method: Finished =======")

    return np.sum(pred == classify_labels)/len(classify_labels)

''' Functions for nonlinear methods '''

def nonlinear_autoencoder(input_size, code_size: int):
    """
    Instanciate and compiles an autoencoder, returns both the autoencoder and just the encoder
    """
    encoder = keras.Sequential([
        keras.layers.Dense(input_size//4, activation='ReLU'),
        keras.layers.Dense(code_size, activation='ReLU'),
    ])
    
    decoder = keras.Sequential([
        keras.layers.Dense(input_size//4, activation='ReLU'),
        keras.layers.Dense(input_size),
    ])
    
    inputs = keras.Input(shape=(input_size,))
    outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs=inputs, outputs=outputs)
    
    autoencoder.compile(optimizer='Adam', loss='MSE')
    return autoencoder, encoder

def nonlinear_autoencoder_with_cnn_encoder(input_width, input_height, input_channels, output_size, code_size: int):
    """
    Instanciate and compiles an autoencoder, returns both the autoencoder and just the encoder
    """
    encoder = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='ReLU'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='ReLU'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(code_size),
    ])
    
    decoder = keras.Sequential([
        keras.layers.Dense(output_size//2, activation='ReLU'),
        keras.layers.Dense(output_size),
    ])
    
    inputs = keras.Input(shape=(input_width, input_height, input_channels))
    outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs=inputs, outputs=outputs)
    
    autoencoder.compile(optimizer='Adam', loss='MSE')
    return autoencoder, encoder

def test_NLE(train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels, 
             n, n_components = 32, verbose=False, train=1, size=28*28):
    if verbose: print("======= Nonlinear ae method: Training and evaluating ... =======")
    if verbose: print("Learning background ...")
    autoencoder, encoder = nonlinear_autoencoder(size, n_components)
    autoencoder.fit(train_images, train_images, epochs=10)

    if verbose: print("Vectorizing ...")
    oneshot_images = encoder(oneshot_images)
    classify_images = encoder(classify_images)

    if verbose: print("Learning oneshot ...")
    nn = min(train, 5)
    neigh = KNeighborsClassifier(n_neighbors = nn)
    neigh.fit(oneshot_images, oneshot_labels)

    if verbose: print("Predicting ...")
    pred = neigh.predict(classify_images)

    if verbose:
        print("Accuracy: ", np.sum(pred == classify_labels)/len(classify_labels))
        print("======= Nonlinear ae method: Finished =======")

    return np.sum(pred == classify_labels)/len(classify_labels)

def test_CNE(train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels, 
             n, n_components = 32, verbose=False, train=1, w=28, h=28, c=1, o=28*28):
    # Attention: since CNN is here, remember to feed in the 2D-shaped image.
    if verbose: print("======= CNN ae method: Training and evaluating ... =======")
    if verbose: print("Learning background ...")
    autoencoder, encoder = nonlinear_autoencoder_with_cnn_encoder(w, h, c, o, n_components)
    autoencoder.fit(train_images, train_images.reshape(-1, w*h*c), epochs=10)

    if verbose: print("Vectorizing ...")
    oneshot_images = encoder(oneshot_images)
    classify_images = encoder(classify_images)

    if verbose: print("Learning oneshot ...")
    nn = min(train, 5)
    neigh = KNeighborsClassifier(n_neighbors = nn)
    neigh.fit(oneshot_images, oneshot_labels)

    if verbose: print("Predicting ...")
    pred = neigh.predict(classify_images)

    if verbose:
        print("Accuracy: ", np.sum(pred == classify_labels)/len(classify_labels))
        print("======= CNN ae method: Finished =======")

    return np.sum(pred == classify_labels)/len(classify_labels)

def convolutional_autoencoder(input_size, code_size: int):
    """
    Instanciate and compiles an autoencoder, returns both the autoencoder and just the encoder

    :param tuple input_size: shape of the input samples
    :param int code_size: size of the new representation space
    :return: autoencoder, encoder
    """
    # YOUR CODE HERE
    encoder = keras.Sequential([
        keras.layers.Conv2D(16, (3,3), padding='same', activation='ReLU'),
        keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
        keras.layers.Conv2D(8, (3,3), padding='same', activation='ReLU'),
        keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
        keras.layers.Conv2D(2, (3,3), padding='same', activation='ReLU'),
        keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(code_size)
    ])
    
    decoder = keras.Sequential([
        keras.layers.Dense(input_size[0] // 4 * input_size[1] // 4 * input_size[2]),
        keras.layers.Reshape(target_shape=(input_size[0] // 4, input_size[1] // 4, input_size[2])),
        keras.layers.Conv2D(8, (3,3), padding='same', activation='ReLU'),
        keras.layers.UpSampling2D((4, 4)),
        keras.layers.Conv2D(1, (3,3), padding='same')
    ])
    
    Input = keras.Input(shape=input_size)
    autoencoder = keras.models.Model(inputs=Input, outputs=decoder(encoder(Input)))
    autoencoder.compile(optimizer='Adam', loss='mse')
    return autoencoder, encoder

def test_CNE2(train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels, 
              n, n_components = 32, verbose=False, train=1, w=28, h=28, c=1, o=28*28):
    # Attention: since CNN is here, remember to feed in the 2D-shaped image.
    if verbose: print("======= CNN-CNN ae method: Training and evaluating ... =======")
    if verbose: print("Learning background ...")
    autoencoder, encoder = convolutional_autoencoder((w, h, c), n_components)
    autoencoder.fit(train_images, train_images, epochs=10)

    if verbose: print("Vectorizing ...")
    oneshot_images = encoder(oneshot_images)
    classify_images = encoder(classify_images)

    if verbose: print("Learning oneshot ...")
    nn = min(train, 5)
    neigh = KNeighborsClassifier(n_neighbors = nn)
    neigh.fit(oneshot_images, oneshot_labels)

    if verbose: print("Predicting ...")
    pred = neigh.predict(classify_images)

    if verbose:
        print("Accuracy: ", np.sum(pred == classify_labels)/len(classify_labels))
        print("======= CNN-CNN ae method: Finished =======")

    return np.sum(pred == classify_labels)/len(classify_labels)