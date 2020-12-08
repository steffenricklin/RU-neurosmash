
# Imports
from convvae import ConvVae
import numpy as np
import PIL
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import Neurosmash

from numpy import asarray
from numpy import savez_compressed
from numpy import load





ip         = "127.0.0.1"
port       = 13000
size       = 128  # 96, 192
timescale  = 1
# agent = Neurosmash.Agent()
# environment = Neurosmash.Environment(ip, port, size, timescale)

# end (true if the episode has ended, false otherwise)
# reward (10 if won, 0 otherwise)
# state (flattened size x size x 3 vector of pixel values)
# The state can be converted into an image as follows:
# image = np.array(state, "uint8").reshape(size, size, 3)
# You can also use to Neurosmash.Environment.state2image(state) function which returns
# the state as a PIL image


def roam_and_collect(rounds=1, save=False):
    data = []
    for r in range(rounds):
        print(f'Starting round {r+1}')
        end, reward, state = environment.reset()
        data.append(state2input(state))
        while not end:
            action = agent.step(end, reward, state)
            end, reward, state = environment.step(action)
            data.append(state2input(state))

    print(f'Done roaming...')

    data = asarray(data)
    print(data.shape)
    if save:
        # save to npz file
        filename = f'rounds{rounds}_size{size}_timescale{timescale}.npz'
        savez_compressed(filename, data)
        print(f'Data saved to file...')
    return data


def load_file(filename):
    # load dict of arrays
    dict_data = load(filename)
    # extract the first array
    data = dict_data['arr_0']
    return data


def initialise_vae(batch_size=10, z_size=32):
    # Create model
    vae = ConvVae(batch_size=batch_size, size=size, z_size=z_size)
    # Initialise weights
    vae.collect_params().initialize(mx.init.Xavier(), ctx=vae.ctx)
    # Activates hybrid-mode of the Hybrid-Block
    vae.hybridize()
    # Initialise trainer
    trainer = gluon.Trainer(vae.collect_params(), 'adam', {'learning_rate': .001})
    return vae, trainer


def state2image(state):
    return np.array(state).reshape(orig_shape).astype(np.uint8)


def state2input(state):
    inpt = np.reshape(state, tens_shape).astype(np.uint8)
    inpt = np.interp(inpt, [0, 255], [0.0, 1.0])
    return inpt


def output2state(output):
    return np.reshape(output.asnumpy(), (-1, *orig_shape))[0]


def refresh_data(init_data, rnd_data, batch_size):
    keep, _ = train_test_split(init_data, test_size=rnd_data.shape[0])
    print(type(keep), keep.shape, type(rnd_data), rnd_data.shape)
    merged = mx.nd.concat(mx.nd.array(keep), mx.nd.array(rnd_data), dim=0)
    new_data = mx.io.NDArrayIter(data={'data': merged}, batch_size=batch_size)
    return new_data


def experience_replay(vae, trainer, init_data, exp_data, train, batch_size, n_epochs, rounds):
    losses = []
    exp_data_rounds = np.array_split(exp_data, rounds)
    for rnd in exp_data_rounds:
        new_data = refresh_data(init_data, rnd, batch_size)
        loss = run_vae(vae, trainer, new_data, train=train, n_epochs=n_epochs)
        losses.append(loss)
    return losses

def run_vae(vae, trainer, data_iter, train=True, n_epochs=50, show_io=False, print_period=1):
    total_loss = []

    for epoch in tqdm(range(n_epochs), desc='epochs'):

        # Reset vars
        epoch_loss = 0
        data_iter.reset()

        n_batch = 0
        for batch in data_iter:
            n_batch += 1
            sample = batch.data.pop().as_in_context(vae.ctx)
            if train:
                with autograd.record():
                    out, loss = vae(sample)
                loss.backward()
                trainer.step(sample.shape[0])
            else:
                out, loss = vae(sample)
            epoch_loss += nd.mean(loss).asscalar()

            if show_io:
                _, axs = plt.subplots(1, 2)
                axs[0].imshow(output2state(sample))
                axs[1].imshow(output2state(out))
                if train:
                    plt.title('Training')
                else:
                    plt.title('Validation')
                plt.show()
                plt.close()

        epoch_loss /= n_batch
        total_loss.append(epoch_loss)

        if epoch % print_period == 0:
            tqdm.write(f'Epoch {epoch}, Loss {epoch_loss:.2f}')
    print("Training done.")

    return total_loss


def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.legend(['train', 'valid'])
    plt.show()


def run(data, batch_size=10, n_epochs=10):

    # remove superfluous data, because of variable batch sizes and input shapes
    end = (data.shape[0] % batch_size)
    if end != 0:
        data = data[:-end]

    print(data.shape)

    # Initialise the VAE
    vae, trainer = initialise_vae(batch_size=batch_size, z_size=32)
    print('VAE initialised...')

    init_data, exp_data = train_test_split(data, test_size=0.33)

    data_iter = mx.io.NDArrayIter(data={'data': init_data}, batch_size=batch_size, shuffle=True)

    # Train the VAE
    training_loss = run_vae(vae, trainer, data_iter, train=True, n_epochs=n_epochs, show_io=True)
    print('VAE trained...')

    losses = experience_replay(vae, trainer, init_data, exp_data, train=True, batch_size=batch_size, n_epochs=n_epochs, rounds=10)


    # Validate the VAE
    # validation_loss = run_vae(
    #     vae, trainer, data, train=False, batch_size=batch_size, n_epochs=n_epochs, show_io=True
    # )
    # print('VAE validated...')

    plt.plot(training_loss)
    # plt.plot(validation_loss)
    plt.show()

    # plot_loss(training_loss, validation_loss)
    # print('Loss plotted...')

    # Save model
    file_name = "net.params"
    vae.save_parameters(file_name)
    print('Model saved...')

    # Load model
    # new_vae = initialise_vae(batch_size=1, z_size=32)
    # new_vae.load_parameters(file_name, ctx=vae.ctx)

    # compare_io(images, outputs)
    # print('comparison done.')


orig_shape = (size, size, 3)
tens_shape = (3, size, size)
if __name__ == '__main__':

    # data = roam_and_collect(rounds=50, save=True)
    # print(f'got data...')
    # print(data.shape)
    filename = f'rounds{50}_size{size}_timescale{1}.npz'
    data = load_file(filename)
    print(data.shape)
    run(data, batch_size=1, n_epochs=1)


















# def dream():
#     # the dreaming
#     # n_samples = 2
#     # dream_states = vae.dream(n_samples=n_samples)
#     # dream_states = dream_states.asnumpy()
#     # dream_samples = np.transpose(dream_states, (0, 2, 3, 1))
#     # plt.imshow(dream_samples[0])
#     # plt.show()
#
#
#     # canvas = np.empty((size * n_samples, size * n_samples, 3))
#     # for i, img in enumerate(dream_samples):
#     #     x = i // n_samples
#     #     y = i % n_samples
#     #     canvas[(n_samples - y - 1) * size:(n_samples - y) * size,
#     #     x * size:(x + 1) * size] = img  # img.reshape(size, size)
#     # plt.figure(figsize=(8, 8))
#     # plt.imshow(canvas, origin="upper", cmap="Greys")
#     # plt.axis('off')
#     # plt.tight_layout()
#     # plt.title("Look at it. How cute, it's dreaming \u2665 ")
#     # plt.show()