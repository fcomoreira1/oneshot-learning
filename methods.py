import emnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

''' Data-handling functions '''

def separate_fewshot(test_images, test_labels, n_shots):
    fewshot_pick = []
    val_pick = []
    for label in np.unique(test_labels):
        for i in np.random.choice(np.where(test_labels == label)[0], n_shots, False):
            fewshot_pick.append(i)
    temp = set(fewshot_pick)
    for i in range(len(test_labels)):
        if not i in temp:
            val_pick.append(i)
    fewshot_images = test_images[fewshot_pick]
    fewshot_labels = test_labels[fewshot_pick]
    val_images = test_images[val_pick]
    val_labels = test_labels[val_pick]
    return fewshot_images, fewshot_labels, val_images, val_labels

''' Extended MNIST related functions '''

# def show_data(train_images, train_labels, oneshot_images, oneshot_labels, validation_images, validation_labels):
#     # works on notebooks with matplotlib inline, at least
#     train_classes = np.unique(np.unique(train_labels))
#     _, axes = plt.subplots(len(train_classes), 5, figsize=(8, 30))
#     for i in range(len(train_classes)):
#         pick = np.where(train_labels == train_classes[i])[0][:5]
#         for j in range(5):
#             axes[i][j].imshow(train_images[pick[j]])
#             axes[i][j].axis('off')
#     plt.title('train data')

def get_emnist(seed, n_train_classes, n_test_classes, ns_shots, verbose=True, reshape=False):
    assert (1 <= n_train_classes <= 45 and n_train_classes + n_test_classes <= 47), 'Invalid choice of n_train_classes and n_test_classes'
    if verbose: print("======= Loading emnist data... =======")
    images, labels = emnist.extract_training_samples('balanced')
    images = images.copy().astype('float') / 255
    if reshape: images = images.reshape(-1, 28 * 28)
    n_test_classes = 47 - n_test_classes

    # divide into train and test
    classes = np.unique(labels)
    np.random.seed(seed)
    np.random.shuffle(classes)
    train_pick = np.where(np.isin(labels, classes[:n_train_classes]))[0]
    test_pick = np.where(np.isin(labels, classes[n_test_classes:]))[0]
    train_images = images[train_pick]
    train_labels = labels[train_pick]
    test_images = images[test_pick]
    test_labels = labels[test_pick]

    # further divide test into fewshot and val
    test_data = [None] * len(ns_shots)
    for i, n_shots in enumerate(ns_shots):
        np.random.seed(seed)
        np.random.shuffle(classes)
        test_data[i] = separate_fewshot(test_images, test_labels, n_shots)
    if verbose: print("======= Finished loading. =======")

    return train_images, train_labels, test_data

# def get_emnist_2(train_classes, test_classes, n_shots, reshape=False, verbose=True):
#     images, labels = emnist.extract_training_samples('balanced')
#     images = images.copy().astype('float') / 255
#     if reshape: images = images.reshape(-1, 28 * 28)

#     # divide into train and test
#     if verbose: print("======= Loading emnist data ... =======")
#     train_pick = np.where(np.isin(labels, train_classes))[0]
#     test_pick = np.where(np.isin(labels, test_classes))[0]
#     train_images = images[train_pick, :, :]
#     train_labels = labels[train_pick]
#     test_images = images[test_pick, :, :]
#     test_labels = labels[test_pick]
#     if verbose:
#         print("Output shapes: ", [i.shape for i in [train_images, train_labels, test_images, test_labels]])
#         print("Train labels: ", np.unique(train_labels))
#         print("Test labels: ", np.unique(test_labels))

#     # further divide test into fewshot and val
#     fewshot_images, fewshot_labels, validation_images, validation_labels = separate_fewshot(test_images, test_labels, n_shots)
#     if verbose: print("======= Finished loading. =======")

#     return train_images, train_labels, fewshot_images, fewshot_labels, validation_images, validation_labels

def train_fewshot(encoder, n_shots, fewshot_images, fewshot_labels):
    return KNeighborsClassifier(n_neighbors=min(n_shots, 5)).fit(encoder(fewshot_images), fewshot_labels)

def train_fewshot_2(encoder, n_shots, fewshot_images, fewshot_labels):
    return KNeighborsClassifier(n_neighbors=min(n_shots, 5)).fit(encoder.transform(fewshot_images), fewshot_labels)

''' Linear methods '''

def test_PCA(seed, n_train_classes, n_test_classes, ns_shots, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False, True)

    if verbose: print("======= PCA method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    pca = PCA(n_components=encoding_size)
    pca.fit(X=train_images)

    accuracies = [None] * len(ns_shots)
    for i, n_shots in enumerate(ns_shots):
        if verbose: print(f'Learning {n_shots}-shot and predicting...')
        fewshot_images, fewshot_labels, val_images, val_labels = test_data[i]
        classifier = train_fewshot_2(pca, n_shots, fewshot_images, fewshot_labels)
        pred = classifier.predict(pca.transform(val_images))
        accuracies[i] = np.sum(pred == val_labels) / val_labels.shape[0]
        if verbose:
            print(f'Accuracy for {n_shots}-shot: {accuracies[i]}')

    if verbose:
        print("======= PCA method: Finished =======")

    return accuracies

def test_LDA(seed, n_train_classes, n_test_classes, ns_shots, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False, True)
    
    if verbose: print("======= LDA method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    lda = LDA(n_components=encoding_size)
    lda.fit(X=train_images, y=train_labels)

    accuracies = [None] * len(ns_shots)
    for i, n_shots in enumerate(ns_shots):
        if verbose: print(f'Learning {n_shots}-shot and predicting...')
        fewshot_images, fewshot_labels, val_images, val_labels = test_data[i]
        classifier = train_fewshot_2(lda, n_shots, fewshot_images, fewshot_labels)
        pred = classifier.predict(lda.transform(val_images))
        accuracies[i] = np.sum(pred == val_labels) / val_labels.shape[0]
        if verbose:
            print(f'Accuracy for {n_shots}-shot: {accuracies[i]}')
    
    if verbose:
        print("======= LDA method: Finished =======")

    return accuracies

''' Nonlinear methods '''

def nonlinear_autoencoder(input_size, code_size: int):
    """
    Instantiates and compiles an autoencoder, returns both the autoencoder and the encoder
    """
    encoder = keras.Sequential([
        keras.layers.Dense(input_size // 4, activation='ReLU'),
        keras.layers.Dense(code_size, activation='ReLU'),
    ])
    
    decoder = keras.Sequential([
        keras.layers.Dense(input_size // 4, activation='ReLU'),
        keras.layers.Dense(input_size),
    ])
    
    inputs = keras.Input(shape=(input_size,))
    outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs=inputs, outputs=outputs)
    
    autoencoder.compile(optimizer='Adam', loss='MSE')
    return autoencoder, encoder

def nonlinear_autoencoder_with_cnn_encoder(input_shape, code_size):
    """
    Instantiates and compiles an autoencoder with CNN encoder, returns both the autoencoder and the CNN encoder
    """
    encoder = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='ReLU'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='ReLU'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(code_size),
    ])
    
    output_size = input_shape[0] * input_shape[1] * input_shape[2]
    decoder = keras.Sequential([
        keras.layers.Dense(output_size // 2, activation='ReLU'),
        keras.layers.Dense(output_size),
    ])
    
    inputs = keras.Input(shape=input_shape)
    outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs=inputs, outputs=outputs)
    
    autoencoder.compile(optimizer='Adam', loss='MSE')
    return autoencoder, encoder

def test_NLAE(seed, n_train_classes, n_test_classes, ns_shots, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False, True)
    
    if verbose: print("======= Nonlinear autoencoder method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    autoencoder, encoder = nonlinear_autoencoder(28 * 28, encoding_size)
    autoencoder.fit(train_images, train_images, epochs=10)

    accuracies = [None] * len(ns_shots)
    for i, n_shots in enumerate(ns_shots):
        if verbose: print(f'Learning {n_shots}-shot and predicting...')
        fewshot_images, fewshot_labels, val_images, val_labels = test_data[i]
        classifier = train_fewshot(encoder, n_shots, fewshot_images, fewshot_labels)
        pred = classifier.predict(encoder(val_images))
        accuracies[i] = np.sum(pred == val_labels) / val_labels.shape[0]
        if verbose:
            print(f'Accuracy for {n_shots}-shot: {accuracies[i]}')
    
    if verbose:
        print("======= Nonlinear autoencoder method: Finished =======")

    return accuracies

def test_NLAE_CNNE(seed, n_train_classes, n_test_classes, ns_shots, encoding_size=32, verbose=True):
    # train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels, 
    #          n, n_components = 32, verbose=False, train=1, w=28, h=28, c=1, o=28*28):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False)

    if verbose: print("======= Nonlinear autoencoder with CNN encoder method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    autoencoder, encoder = nonlinear_autoencoder_with_cnn_encoder((28, 28, 1), encoding_size)
    autoencoder.fit(train_images, train_images.reshape(-1, 28 * 28), epochs=10)

    accuracies = [None] * len(ns_shots)
    for i, n_shots in enumerate(ns_shots):
        if verbose: print(f'Learning {n_shots}-shot and predicting...')
        fewshot_images, fewshot_labels, val_images, val_labels = test_data[i]
        classifier = train_fewshot(encoder, n_shots, fewshot_images, fewshot_labels)
        pred = classifier.predict(encoder(val_images))
        accuracies[i] = np.sum(pred == val_labels) / val_labels.shape[0]
        if verbose:
            print(f'Accuracy for {n_shots}-shot: {accuracies[i]}')
    
    if verbose:
        print("======= Nonlinear autoencoder method: Finished =======")

    return accuracies

# def convolutional_autoencoder(input_size, code_size: int):
#     """
#     Instanciate and compiles an autoencoder, returns both the autoencoder and just the encoder

#     :param tuple input_size: shape of the input samples
#     :param int code_size: size of the new representation space
#     :return: autoencoder, encoder
#     """
#     # YOUR CODE HERE
#     encoder = keras.Sequential([
#         keras.layers.Conv2D(16, (3,3), padding='same', activation='ReLU'),
#         keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
#         keras.layers.Conv2D(8, (3,3), padding='same', activation='ReLU'),
#         keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
#         keras.layers.Conv2D(2, (3,3), padding='same', activation='ReLU'),
#         keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
#         keras.layers.Flatten(),
#         keras.layers.Dense(code_size)
#     ])
    
#     decoder = keras.Sequential([
#         keras.layers.Dense(input_size[0] // 4 * input_size[1] // 4 * input_size[2]),
#         keras.layers.Reshape(target_shape=(input_size[0] // 4, input_size[1] // 4, input_size[2])),
#         keras.layers.Conv2D(8, (3,3), padding='same', activation='ReLU'),
#         keras.layers.UpSampling2D((4, 4)),
#         keras.layers.Conv2D(1, (3,3), padding='same')
#     ])
    
#     Input = keras.Input(shape=input_size)
#     autoencoder = keras.models.Model(inputs=Input, outputs=decoder(encoder(Input)))
#     autoencoder.compile(optimizer='Adam', loss='mse')
#     return autoencoder, encoder

# def test_CNE2(train_images, train_labels, oneshot_images, oneshot_labels, classify_images, classify_labels, 
#               n, n_components = 32, verbose=False, train=1, w=28, h=28, c=1, o=28*28):
#     # Attention: since CNN is here, remember to feed in the 2D-shaped image.
#     if verbose: print("======= CNN-CNN ae method: Training and evaluating ... =======")
#     if verbose: print("Learning background ...")
#     autoencoder, encoder = convolutional_autoencoder((w, h, c), n_components)
#     autoencoder.fit(train_images, train_images, epochs=10)

#     if verbose: print("Vectorizing ...")
#     oneshot_images = encoder(oneshot_images)
#     classify_images = encoder(classify_images)

#     if verbose: print("Learning oneshot ...")
#     nn = min(train, 5)
#     neigh = KNeighborsClassifier(n_neighbors = nn)
#     neigh.fit(oneshot_images, oneshot_labels)

#     if verbose: print("Predicting ...")
#     pred = neigh.predict(classify_images)

#     if verbose:
#         print("Accuracy: ", np.sum(pred == classify_labels)/len(classify_labels))
#         print("======= CNN-CNN ae method: Finished =======")

#     return np.sum(pred == classify_labels)/len(classify_labels)