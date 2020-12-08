
# Imports
from convvae import ConvVae
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import Neurosmash

ip         = "127.0.0.1"
port       = 13000
size       = 128  # 96, 192
timescale  = 1
agent = Neurosmash.Agent()
environment = Neurosmash.Environment(ip, port, size, timescale)

# end (true if the episode has ended, false otherwise)
# reward (10 if won, 0 otherwise)
# state (flattened size x size x 3 vector of pixel values)
# The state can be converted into an image as follows:
# image = np.array(state, "uint8").reshape(size, size, 3)
# You can also use to Neurosmash.Environment.state2image(state) function which returns
# the state as a PIL image

shape = (3, size, size)


def roam_and_collect(nr_images=10):

    # The following steps through an entire episode from start to finish with random actions (by default)
    images = np.zeros((nr_images, *shape))

    # Do as many rounds as we need to collect the desired number of images
    i = 0
    while i < nr_images:
        end, reward, state = environment.reset()
        # While no one is dead, roam the environment and collect images
        while end == 0:
            action = agent.step(end, reward, state)
            end, reward, state = environment.step(action)
            images[i] = np.array(state).reshape(shape).astype(np.uint8)
            i += 1
            if i >= nr_images:
                end = 1

    return images


def plot_samples(images, n_samples=10):
    idx = np.random.choice(images.shape[0], n_samples)
    _, axarr = plt.subplots(1, n_samples, figsize=(16, 4))
    for i, j in enumerate(idx):
        axarr[i].imshow(images[j].reshape(shape[1], shape[2], shape[0]).astype(np.uint8))
        axarr[i].get_xaxis().set_ticks([])
        axarr[i].get_yaxis().set_ticks([])
    plt.show()


def initialise_vae(data, batch_size=10, z_size=32):

    # Split data
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)  # change random state to get different distribution
    train_iter = mx.io.NDArrayIter(data={'data': train_data}, batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(data={'data': test_data}, batch_size=batch_size)

    # Create model
    vae = ConvVae(batch_size=batch_size, size=size, z_size=z_size)
    vae.collect_params().initialize(mx.init.Xavier(), ctx=vae.ctx)
    vae.hybridize()  # activates hybrid-mode of the Hybrid-Block
    trainer = gluon.Trainer(vae.collect_params(), 'adam', {'learning_rate': .001})

    return vae, train_iter, test_iter, trainer


def train_vae(vae, train_iter, test_iter, trainer, n_epochs=50, print_period=10):
    training_loss = []
    validation_loss = []
    outputs = np.zeros((1, *shape))

    for epoch in tqdm(range(n_epochs), desc='epochs'):

        # Reset vars
        epoch_loss = 0
        train_iter.reset()
        test_iter.reset()

        # Training data
        n_batch_train = 0
        for batch in train_iter:
            n_batch_train += 1
            data = batch.data.pop().as_in_context(vae.ctx)
            with autograd.record():
                out, loss = vae(data)
            loss.backward()
            trainer.step(data.shape[0])
            epoch_loss += nd.mean(loss).asscalar()

            # See reconstruction
            plt.imshow(np.reshape(out[0].asnumpy(), (size, size, 3)))
            plt.show()

        # Validation data
        n_batch_val = 0
        epoch_val_loss = 0
        for batch in test_iter:
            n_batch_val += 1
            data = batch.data.pop().as_in_context(vae.ctx)
            out, loss = vae(data)
            epoch_val_loss += nd.mean(loss).asscalar()
            outputs = np.concatenate((outputs, out.asnumpy()), axis=0)

        epoch_loss /= n_batch_train
        epoch_val_loss /= n_batch_val

        training_loss.append(epoch_loss)
        validation_loss.append(epoch_val_loss)

        if epoch % max(print_period, 1) == 0:
            tqdm.write(
                'Epoch{}, Training loss {:.2f}, Validation loss {:.2f}'.format(epoch, epoch_loss, epoch_val_loss))
    print("done.")

    return training_loss, validation_loss, outputs


def plot_reconstruction(out):
    output = np.reshape(out, (size, size, 3))
    plt.imshow(output)
    plt.show()


def plot_loss(n_epochs, training_loss, validation_loss):
    batch_x = np.linspace(1, n_epochs, len(training_loss))
    plt.plot(batch_x, np.array(training_loss))
    plt.plot(batch_x, np.array(validation_loss))
    plt.legend(['train', 'valid'])
    plt.show()


def compare_io(images, outputs):
    # Original
    plt.figure()
    plt.imshow(images[-1].reshape(shape[1], shape[2], shape[0]).astype(np.uint8))
    plt.title('Original')
    plt.show()

    # Reconstruction
    # rnd = np.random.randint(outputs.shape[0])
    plot_reconstruction(outputs[-1])


def run(nr_images=100, n_epochs=10):
    images = roam_and_collect(nr_images=nr_images)
    print('got images...')
    plot_samples(images, n_samples=10)
    print('plotted samples...')
    # map images to the range [0, 1]
    data = np.interp(images, [0, 255], [0.0, 1.0])
    vae, train_iter, test_iter, trainer = initialise_vae(data, batch_size=10, z_size=32)
    print('vae initialised...')
    training_loss, validation_loss, outputs = train_vae(vae, train_iter, test_iter, trainer, n_epochs, print_period=1)
    print('vae trained...')
    plot_loss(n_epochs, training_loss, validation_loss)
    print('loss plotted...')
    compare_io(images, outputs)
    print('comparison done.')


if __name__ == '__main__':
    run(250, 50)
