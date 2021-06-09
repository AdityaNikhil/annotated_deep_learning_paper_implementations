import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa



class params:

    learning_rate = 0.003
    weight_decay = 0.1
    batch_size = 512 
    num_epochs = 7 
    image_size = 72  # We'll resize input images to this size
    patch_size = 16  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 128
    num_heads = 8
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 10 #Changed from 13 to 10
    mlp_head_units = [3072, 768]  # Size of the dense layers of the final classifier


def load_data(dataset):

    if dataset=='CIFAR10':

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        num_classes = 10

        return (x_train, y_train), (x_test, y_test), num_classes 
    if dataset=='CIFAR100':

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        num_classes = 100

        return (x_train, y_train), (x_test, y_test), num_classes
    if dataset=='MNIST':

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        num_classes = 10

        return (x_train, y_train), (x_test, y_test), num_classes             


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        # Flattening patches
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):

        # config = super().get_config().copy()
        config= {
            'patch_size': self.patch_size,
        }
        
        return config    


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        # config = super().get_config().copy()
        config= {
            'num_patches': self.num_patches,
            'projection' : self.projection,
            'position_embedding' : self.position_embedding,
        }
        
        return config         


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output        



def create_vit_classifier(num_classes):
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(72, 72),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )    
    # Parameters
    configs = params
    # model inputs
    inputs = layers.Input(shape=(32, 32, 3))
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(configs.patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(configs.num_patches, configs.projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(configs.transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadSelfAttention(
            num_heads=configs.num_heads, embed_dim=configs.projection_dim
        )(x1)  
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=configs.transformer_units, dropout_rate=0.2)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=configs.mlp_head_units, dropout_rate=0.5)  
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model



def run_experiment(model, xtrain, ytrain, xtest=None, ytest=None, model_save=False):
    configs = params

    optimizer = tfa.optimizers.AdamW(
        learning_rate=configs.learning_rate, weight_decay=configs.weight_decay, beta_1=0.9, beta_2=0.999
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    if xtest is None:
        history = model.fit(
            x=xtrain,
            y=ytrain,
            batch_size=configs.batch_size,
            epochs=configs.num_epochs,
            # callbacks=[wandb_callback],
        ) 

    else:    
        history = model.fit(
            x=xtrain,
            y=ytrain,
            batch_size=configs.batch_size,
            epochs=configs.num_epochs,
            validation_split=0.1,
            validation_data = (xtest, ytest)
            # callbacks=[wandb_callback],
        )   

    if model_save:
        model.save('ViT_model.h5')

    return history



