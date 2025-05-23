import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from geometry import TIGREDataset
from todo import *
import skimage.io
# TODO CURRENTLY BETTER THAN PT2 UP TO AT LEAST 18!!
# NOTE: The hyperparameter values in this file are set to similar numbers to the NAF paper.
# You are encouraged to experiment and change them to find something that works better
# for your architectural change. These should work fine for Step 1.
class ResDense(tf.keras.layers.Layer):
    def __init__(self, in_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(in_dim)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.activation(x)
        x = tf.concat([inputs, x], axis=-1)
        x = self.norm(x)
        return x

class AttentionBlock(tf.keras.layers.Layer):
    '''
    Based off of Attention Is All You Need Vaswani et al.
    mhattention->add/norm->dense->add/norm

    '''
    def __init__(self, att_dim, heads, mlp_dim):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=att_dim//heads)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(24, activation=tf.keras.layers.LeakyReLU(0.2))
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        attn = self.attention(inputs, inputs)
        x = self.norm1(inputs + attn)
        dense_out = self.dense(x)
        return self.norm2(x + dense_out)
        return x


class Model(tf.keras.layers.Layer):
    """
    A model class for Attenuation coefficient prediction from https://arxiv.org/abs/2209.14540
    This implementation uses an argument encoder to encode points in 3-dimensional space and
    then passes the encoding to several dense layers to produce the predicted attenuation
    at that point in 3-dimensional space.
    """

    def __init__(self, encoder, bound=0.2, num_layers=3, hidden_dim=32, skips=(1, 3), out_dim=1,
                 last_activation="sigmoid"):
        super(Model, self).__init__()

        self.encoder = encoder
        self.bound = bound
        self.in_dim = self.encoder.get_output_dim()  # Get the input dimension from the encoder
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.out_dim = out_dim

        # Define the layers
        self.layers = []
        # First layer
        self.attention = AttentionBlock(self.in_dim, 2, self.hidden_dim)

        self.layers.append(tf.keras.layers.Dense(hidden_dim))

        # Intermediate layers
        for i in range(1, num_layers - 1):

            # TODO: maybe try embed -> FF -> attention -> FF -> attention ...
            #  try with and without resi
            #  maybe get rid of skips bc bad vibes

            if i in skips:
                self.layers.append(ResDense(hidden_dim + self.in_dim))
            else:
                self.layers.append(ResDense(hidden_dim))

        # Output layer
        self.layers.append(tf.keras.layers.Dense(out_dim))

        # Activation functions
        self.activations = []
        # self.norms = []
        for i in range(num_layers - 1):
            self.activations.append(tf.keras.layers.LeakyReLU(alpha=0.2))  # Equivalent to nn.LeakyReLU() in PyTorch
            # self.norms.append(tf.keras.layers.Normalization())

        # Handle last activation
        if last_activation == "sigmoid":
            self.activations.append(tf.keras.layers.Activation("sigmoid"))
        elif last_activation == "relu":
            self.activations.append(tf.keras.layers.LeakyReLU(alpha=0.2))
        else:
            raise NotImplementedError("Unknown last activation")

    def call(self, x):
        # First, encode the input using the encoder
        # print(x.shape)
        n_rays, n_points = x.shape[0], x.shape[1]
        x = tf.reshape(x, (-1, 3))
        x = self.encoder(x)
        # n_rays = tf.shape(x)[0] // n_points  # assuming 192 n_points; make configurable
        x = tf.reshape(x, (n_rays, n_points, -1))  # [B, T, D]
        # print(x.shape)

        # Step 3: Attention block(s)
        x = self.attention(x)

        # Step 4: Collapse sequence if needed
        x = tf.reshape(x, (-1, x.shape[-1]))  # [B*T, D]
        # Extract input points (if needed for skip connections)
        input_pts = x[..., :self.in_dim]
        # x = tf.reshape(x_encode, [x.shape[0], x.shape[1], -1])  # [batch, seq_len, in_dim]
        # x = self.attention(x)

        # Apply the layers
        for i in range(self.num_layers):
            layer = self.layers[i]
            activation = self.activations[i] if i < len(self.activations) else None
            # norm = self.norms[i] if i < len(self.norms) else None

            # If this layer is a skip layer, concatenate the input points
            if i in self.skips:
                x = tf.concat([input_pts, x], axis=-1)

            # Apply the linear transformation
            x = layer(x)

            # Apply the activation function
            if activation:
                x = activation(x)
            # if norm:
            #     x = norm(x)

        return x

def train(model, dataset, optimizer, n_points):
    """
    Simple training loop that iterates through each projection image, samples rays from that image,
    sends the points of those rays through the network, computes the predicted attenuation per ray,
    computes the loss between the predicted value and the true value, and then updates the network.
    """
    num_projections = dataset.rays.shape[-1]
    total_loss = 0
    for i in range(num_projections):
        projection, rays = dataset[i]
        all_points, all_distances = rays_to_points(rays, n_points, dataset.near, dataset.far)
        # print(all_points.shape)
        # print(all_points.shape)
        magnitudes = tf.norm(rays[..., 3:6], axis=-1)
        n_rays = all_points.shape[0]
        points = tf.reshape(all_points, (-1, 3))

        for start in range(0, n_rays, n_rays):
            end = min(start + n_rays, n_rays)

            # points = tf.reshape(all_points[start:end], (-1, 3))
            distances = all_distances
            mags = magnitudes[start:end]
            proj = projection[start:end]

            with tf.GradientTape() as tape:
                attenuation = model(all_points)
                attenuation = tf.reshape(attenuation, (end-start, -1))
                predicted_attenuation = ray_attenuation(attenuation, distances, mags, dataset.near, dataset.far)


                loss = tf.keras.losses.MSE(projection, predicted_attenuation)
                total_loss += loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss / num_projections


def get_sample_slices(model, dataset):
    """
    Queries the network at many 3-dimensional points to produce a voxelized grid of attenuation values.
    Note that the returned values are scaled by the max value.
    """
    slices = dataset.voxels.shape[2]
    slice_list = []
    for i in range(slices):
        voxels = tf.convert_to_tensor(dataset.voxels[:, :, i])
        shape = voxels.shape[0:2]
        # voxels = tf.reshape(voxels, (-1, 3))

        image_pred = model(voxels)
        image_pred = tf.reshape(image_pred, shape)
        slice_list.append(image_pred)

    imarr = np.array(slice_list)
    imarr = ((imarr / np.max(imarr)) * 255).astype(np.uint8)
    return imarr

def main(dataset_path, epochs, n_points, n_rays):
    """
    Loads the data, saves a ground truth image, and then creates the model.
    Runs for a given number of epochs and number of sample points/sample rays for each projection image training loop.
    Saves a TIFF image of the sample slice output every 10 epochs.
    """
    dataset = TIGREDataset(dataset_path, device="cuda", n_rays=n_rays)

    # need to transpose to get top down view
    ground_truth_volume = (dataset.ground_truth.transpose((2,0,1))*255).astype(np.uint8)

    skimage.io.imsave(f'data/out/gt.tiff', ground_truth_volume)

    size = dataset.far - dataset.near
    encoder = PositionEmbeddingEncoder(size, 8, 3, 3)
    model = Model(encoder)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    print(f'Starting training...')
    for epoch in range(epochs):
        epoch_loss = train(model, dataset, optimizer, n_points)

        print(f'Epoch {epoch} loss: {epoch_loss}')

        if epoch % 10 == 0 and epoch != 0:
            predicted_volume = get_sample_slices(model, dataset)
            print(f'Epoch {epoch} SSIM: {tf.image.ssim(predicted_volume, ground_truth_volume, max_val=255)}'+ \
                f' PSNR {tf.image.psnr(predicted_volume, ground_truth_volume, max_val=255)}')

            if not os.path.exists('data/out/'):
                os.mkdir('data/out/')

            skimage.io.imsave(f'data/out/{epoch}.tiff', predicted_volume)


if __name__ == '__main__':
    dataset_path = 'data/ct_data/chest_50.pickle'
    # dataset_path = 'data/ct_data/abdomen_50.pickle'
    # dataset_path = 'data/ct_data/foot_50.pickle'
    # dataset_path = 'data/ct_data/jaw_50.pickle'

    # 250 epochs is not enough to produce a high quality reconstruction but you should see
    # a clear shape after 10 epochs
    epochs = 251
    main(dataset_path, epochs=epochs, n_points=192, n_rays=2048)
