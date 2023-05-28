import emnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Activation, Dropout
from tensorflow.keras.models import Sequential, Model

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
    _, w, h = images.shape
    if reshape: images = images.reshape(-1, w * h)
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

''' Auxiliary functions '''

def train_fewshot(encoder, n_shots, fewshot_images, fewshot_labels):
    return KNeighborsClassifier(n_neighbors=min(n_shots, 5)).fit(encoder(fewshot_images), fewshot_labels)

def trains_fewshot(encoder, ns_shots, test_data, verbose=True):
    accuracies = [None] * len(ns_shots)
    for i, n_shots in enumerate(ns_shots):
        if verbose: print(f'Learning {n_shots}-shot and predicting...')
        fewshot_images, fewshot_labels, val_images, val_labels = test_data[i]
        classifier = train_fewshot(encoder, n_shots, fewshot_images, fewshot_labels)
        pred = classifier.predict(encoder(val_images))
        accuracies[i] = np.sum(pred == val_labels) / val_labels.shape[0]
        if verbose:
            print(f'Accuracy for {n_shots}-shot: {accuracies[i]}')
    return accuracies

def train_fewshot_2(encoder, n_shots, fewshot_images, fewshot_labels):
    return KNeighborsClassifier(n_neighbors=min(n_shots, 5)).fit(encoder.transform(fewshot_images), fewshot_labels)

def trains_fewshot_2(encoder, ns_shots, test_data, verbose=True):
    accuracies = [None] * len(ns_shots)
    for i, n_shots in enumerate(ns_shots):
        if verbose: print(f'Learning {n_shots}-shot and predicting...')
        fewshot_images, fewshot_labels, val_images, val_labels = test_data[i]
        classifier = train_fewshot_2(encoder, n_shots, fewshot_images, fewshot_labels)
        pred = classifier.predict(encoder.transform(val_images))
        accuracies[i] = np.sum(pred == val_labels) / val_labels.shape[0]
        if verbose:
            print(f'Accuracy for {n_shots}-shot: {accuracies[i]}')
    return accuracies

''' Linear methods '''

def test_PCA(seed, n_train_classes, n_test_classes, ns_shots, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False, True)

    if verbose: print("======= PCA method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    pca = PCA(n_components=encoding_size)
    pca.fit(X=train_images)

    accuracies = trains_fewshot_2(pca, ns_shots, test_data)

    if verbose:
        print("======= PCA method: Finished =======")

    return accuracies

def test_LDA(seed, n_train_classes, n_test_classes, ns_shots, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False, True)
    
    if verbose: print("======= LDA method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    lda = LDA(n_components=encoding_size)
    lda.fit(X=train_images, y=train_labels)

    accuracies = trains_fewshot_2(lda, ns_shots, test_data)
    
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
    
    autoencoder.compile(optimizer=optimizers.legacy.Adam(), loss='MSE')
    return autoencoder, encoder

def cnn_encoder(w, h, encoding_size):
    return Sequential([
        Conv2D(16, (3, 3), input_shape=(w, h, 1), activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=2, strides=(2, 2)),
        Dropout(0.25),

        Conv2D(32, (3, 3), kernel_regularizer='l2'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=2, strides=(2, 2)),
        Dropout(0.25),

        Flatten(),
        
        Dense(encoding_size),
    ])

def nonlinear_autoencoder_with_cnn_encoder(input_shape, code_size):
    """
    Instantiates and compiles an autoencoder with CNN encoder, returns both the autoencoder and the CNN encoder
    """
    encoder = cnn_encoder(input_shape[0], input_shape[1], code_size)
    
    output_size = input_shape[0] * input_shape[1] * input_shape[2]
    decoder = keras.Sequential([
        keras.layers.Dense(output_size // 2, activation='ReLU'),
        keras.layers.Dense(output_size),
    ])
    
    inputs = keras.Input(shape=input_shape)
    outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs=inputs, outputs=outputs)
    
    autoencoder.compile(optimizer=optimizers.legacy.Adam(), loss='MSE')
    return autoencoder, encoder

def test_NLAE(seed, n_train_classes, n_test_classes, ns_shots, n_epochs=15, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False, True)
    
    if verbose: print("======= Nonlinear autoencoder method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    autoencoder, encoder = nonlinear_autoencoder(train_images.shape[1], encoding_size)
    autoencoder.fit(train_images, train_images, epochs=n_epochs)

    accuracies = trains_fewshot(encoder, ns_shots, test_data)
    
    if verbose:
        print("======= Nonlinear autoencoder method: Finished =======")

    return accuracies

def test_NLAE_CNNE(seed, n_train_classes, n_test_classes, ns_shots, n_epochs=15, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False)

    if verbose: print("======= Nonlinear autoencoder with CNN encoder method: Training and evaluating... =======")
    if verbose: print("Learning background...")
    _, w, h = train_images.shape
    autoencoder, encoder = nonlinear_autoencoder_with_cnn_encoder((w, h, 1), encoding_size)
    autoencoder.fit(train_images, train_images.reshape(-1, w * h), epochs=n_epochs)

    accuracies = trains_fewshot(encoder, ns_shots, test_data)
    
    if verbose:
        print("======= Nonlinear autoencoder with CNN encoder method: Finished =======")

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

def get_image_by_label(train_images, train_labels, label):
    return train_images[np.random.choice(np.where(train_labels == label)[0], 1, replace=False)[0]]

def get_train_data(train_images, train_labels, batch_size):
    _, w, h = train_images.shape
    targets = np.zeros((batch_size,))
    targets[batch_size // 2:] = 1
    pairs = [np.zeros((batch_size, w, h)) for _ in range(2)]
    labels = np.unique(train_labels)
    for i in range(batch_size):
        class1, class2 = np.random.choice(labels, 2, replace=False)
        assert(class1 != class2)
        pairs[0][i] = get_image_by_label(train_images, train_labels, class1)
        pairs[1][i] = get_image_by_label(train_images, train_labels, class2 if i < batch_size // 2 else class1)
    return pairs, targets

def get_triplets(train_images, train_labels, batch_size, w, h):
    triplets = [np.zeros((batch_size, w, h)) for _ in range(3)]
    labels = np.unique(train_labels)
    for i in range(batch_size):
        class1, class2 = np.random.choice(labels, 2, replace=False)
        assert(class1 != class2)
        triplets[0][i] = get_image_by_label(train_images, train_labels, class1)
        triplets[1][i] = get_image_by_label(train_images, train_labels, class1)
        triplets[2][i] = get_image_by_label(train_images, train_labels, class2)
    return triplets

def siamese_net(w, h, encoding_size):
    left_input = Input((w, h, 1))
    right_input = Input((w, h, 1))
    
    encoder = cnn_encoder(w, h, encoding_size)
    
    left_emb = encoder(left_input)
    right_emb = encoder(right_input)
    
    L1_Layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_Dist = L1_Layer([left_emb,right_emb])
    OP = Dense(1, activation='sigmoid', kernel_regularizer='l2')(L1_Dist)
    
    siamese_network = Model(inputs=[left_input, right_input], outputs=OP)
    
    return siamese_network, encoder

def test_SN(seed, n_train_classes, n_test_classes, ns_shots, n_iterations=3000, batch_size=10, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False)

    _, w, h = train_images.shape

    if verbose: print("======= Siamese network method: Training and evaluating... =======")
    if verbose: print("Learning background...")

    siamese_network, encoder = siamese_net(w, h, encoding_size)
    siamese_network.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.legacy.Adam(),
        metrics=['accuracy']
    )

    for i in range(n_iterations):
        x, y = get_train_data(train_images, train_labels, batch_size)
        siamese_network.fit(x, y, verbose=False)

    accuracies = trains_fewshot(encoder, ns_shots, test_data)
    
    if verbose:
        print("======= Siamese network method: Finished =======")

    return accuracies

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

def siamese_net_for_triplet_loss(w, h, encoding_size):
    anchor_input = layers.Input(name="anchor", shape=(w, h, 1))
    positive_input = layers.Input(name="positive", shape=(w, h, 1))
    negative_input = layers.Input(name="negative", shape=(w, h, 1))

    encoder = cnn_encoder(w, h, encoding_size)

    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return siamese_network, encoder

class SiameseModel(Model):
    """
    The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "pdistance": self.siamese_network(data)[0], "ndistance": self.siamese_network(data)[1]}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

def test_SN_TL(seed, n_train_classes, n_test_classes, ns_shots, n_iterations=2000, batch_size=10, encoding_size=32, verbose=True):
    train_images, train_labels, test_data = get_emnist(seed, n_train_classes, n_test_classes, ns_shots, False)

    _, w, h = train_images.shape

    if verbose: print("======= Siamese network with triplet loss method: Training and evaluating... =======")
    if verbose: print("Learning background...")

    siamese_network, encoder = siamese_net_for_triplet_loss(w, h, encoding_size)
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.legacy.Adam())
    for _ in range(n_iterations):
        siamese_model.fit(get_triplets(train_images, train_labels, batch_size, w, h), verbose=False)

    accuracies = trains_fewshot(encoder, ns_shots, test_data)
    
    if verbose:
        print("======= Siamese network with triplet loss method: Finished =======")

    return accuracies