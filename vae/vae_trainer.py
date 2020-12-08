
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

ip         = "127.0.0.1"
port       = 13000
size       = 128  # 96, 192
timescale  = 10
agent = Neurosmash.Agent()
environment = Neurosmash.Environment(ip, port, size, timescale)

# end (true if the episode has ended, false otherwise)
# reward (10 if won, 0 otherwise)
# state (flattened size x size x 3 vector of pixel values)
# The state can be converted into an image as follows:
# image = np.array(state, "uint8").reshape(size, size, 3)
# You can also use to Neurosmash.Environment.state2image(state) function which returns
# the state as a PIL image


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
    inpt = np.reshape(state, (1, *tens_shape)).astype(np.uint8)
    inpt = np.interp(inpt, [0, 255], [0.0, 1.0])
    inpt = mx.nd.array(inpt)
    return inpt

def output2state(output):
    return np.reshape(output.asnumpy(), (1, *orig_shape))[0]

def run_vae(vae, trainer, train=True, n_epochs=50, n_images=100, show_input=False, show_output=False):

    total_loss = []

    ### Training
    print('Starting training...')
    i = 0
    while i < n_images:

        end, reward, state = environment.reset()

        while not end:

            # Find an action
            action = agent.step(end, reward, state)

            # Take a step; get a new state
            end, reward, state_next = environment.step(action)

            # If nothing changes anymore, break
            if state == state_next:
                print('resetting environment')
                break

            # Save next state
            state = state_next

            if show_input:
                # show input
                plt.imshow(state2image(state))
                plt.show()
                plt.close()

            # Train on one image
            data = state2input(state)
            epoch_loss = 0
            for _ in range(n_epochs):

                if train:
                    with autograd.record():
                        out, loss = vae(data)
                    loss.backward()
                    trainer.step(1)  # batch size = 1
                else:
                    out, loss = vae(data)

                epoch_loss += loss.asscalar()

            total_loss.append(epoch_loss/n_epochs)

            print(f'round {i} with loss {epoch_loss/n_epochs}')

            if show_output:
                # Plot reconstruction
                state_again = output2state(out)
                plt.imshow(state_again)
                plt.show()
                plt.close()

            # Stop if we have the desired amount of images
            i += 1
            if i >= n_images:
                end = 1

    return total_loss


def plot_reconstruction(out):
    output = np.reshape(out, (size, size, 3))
    plt.imshow(output)
    plt.show()


def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.legend(['train', 'valid'])
    plt.show()


def run(n_images_train=100, n_images_valid=100, n_epochs=10):

    # Initialise the VAE
    vae, trainer = initialise_vae(batch_size=1, z_size=32)
    print('VAE initialised...')

    # Train the VAE
    training_loss = run_vae(
        vae, trainer, train=True, n_epochs=n_epochs, n_images=n_images_train, show_input=False, show_output=False
    )
    print('VAE trained...')

    # Validate the VAE
    validation_loss = run_vae(
        vae, trainer, train=False, n_epochs=n_epochs, n_images=n_images_valid, show_input=False, show_output=False
    )
    print('VAE validated...')

    plot_loss(training_loss, validation_loss)
    print('Loss plotted...')

    # Save model
    file_name = "net.params"
    vae.save_parameters(file_name)
    print('Model saved...')

    # Load model
    # new_vae = initialise_vae(batch_size=1, z_size=32)
    # new_vae.load_parameters(file_name, ctx=vae.ctx)

    # compare_io(images, outputs)
    # print('comparison done.')



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

orig_shape = (size, size, 3)
tens_shape = (3, size, size)
if __name__ == '__main__':
    run(n_images_train=250, n_images_valid=100, n_epochs=1)
