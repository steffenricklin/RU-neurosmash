# Imports
from vae.convvae import ConvVae
import mxnet as mx
from mxnet import gluon, autograd
from src.background_extractor import *
import mxnet.ndarray as nd

ip         = "127.0.0.1"
port       = 13000
size       = 128  # 96, 192
timescale  = 1
agent = Neurosmash.Agent()
environment = Neurosmash.Environment(ip, port, size, timescale)

def initialise_vae(batch_size=10, z_size=32):
    # Create model
    vae = ConvVae(batch_size=batch_size, size=size, z_size=z_size)
    # Initialise weigh
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

def remove_background(state, background):
    cleaned_state = np.where(state == background, 100, state)
    return cleaned_state

def train_vae(vae, trainer, n_epochs=50, print_period=10, n_images_train=100, n_images_valid=100, n_init_rounds = 5):

    training_loss = []
    validation_loss = []

    extr = Background_Extractor()
    background = np.reshape(extr.get_background(size),(1,-1))

    print("Creating initial data")
    buffer = nd.zeros((n_init_rounds*500, size*size*3))
    end, reward, state = environment.reset()
    buffered_images = 0
    rounds = 0
    while buffered_images < n_init_rounds*500 and rounds < 5:
        while end == 0:
            action = agent.step(end, reward, state)
            end, reward, state = environment.step(action)
            buffer[buffered_images] = state
            buffered_images += 1
            if buffered_images < n_init_rounds*500:
                break
        rounds += 1



    ### Training
    print('Starting training...')
    i = 0
    while i < n_images_train:
        print('New battle in training mode...')
        end, reward, state = environment.reset()

        while not end:
            action = agent.step(end, reward, state)
            end, reward, state_next = environment.step(action)
            if state == state_next:
                print('resetting environment')
                break
            state = state_next
            state_no_background = remove_background(state, background)
            data = state2input(state_no_background)
            epoch_loss = 0
            for _ in range(n_epochs):
                with autograd.record():
                    out, loss = vae(data)
                loss.backward()
                trainer.step(1)  # batch size = 1
                print(epoch_loss)

            training_loss.append(epoch_loss/n_epochs)
            print(f'training round {i} with loss {epoch_loss/n_epochs}')

            # Plot training reconstruction
            stateagain = output2state(out)
            plt.imshow(stateagain)
            plt.show()

            # Stop if we have the desired amount of images
            i += 1
            if i >= n_images_train:
                end = 1


    ### Validation
    print('Starting validation...')

    i = 0
    while i < n_images_valid:
        print('New battle in validation mode...')
        end, reward, state = environment.reset()

        while not end:

            # Find an action
            action = agent.step(end, reward, state)

            # Take a step; get a new state
            end, reward, state_next = environment.step(action)

            # If nothing changes anymore, presumably when an agent has died, break
            if state == state_next:
                print('resetting environment')
                break

            # Save next state
            state = state_next

            data = state2input(state)

            epoch_val_loss = 0
            for _ in range(n_epochs):
                out, loss = vae(data)
                print(f'validation loss: {loss}')
                epoch_val_loss += loss.asscalar()
            validation_loss.append(epoch_val_loss/n_epochs)



            # Plot validation reconstruction
            print(f'validation reconstruction after {i} images')
            stateagain = output2state(out)
            plt.imshow(stateagain)
            plt.show()

            # Stop if we have the desired amount of images
            i += 1
            if i >= n_images_train:
                end = 1

    return training_loss, validation_loss


def plot_reconstruction(out):
    output = np.reshape(out, (size, size, 3))
    plt.imshow(output)
    plt.show()


def plot_loss(n_epochs, training_loss, validation_loss):
    # batch_x = np.linspace(1, n_epochs, len(training_loss))
    # plt.plot(batch_x, np.array(training_loss))
    # plt.plot(batch_x, np.array(validation_loss))
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.legend(['train', 'valid'])
    plt.show()


def run(n_images_train=100, n_images_valid=100, n_epochs=10):

    # Initialise the VAE
    vae, trainer = initialise_vae(batch_size=1, z_size=32)
    print('VAE initialised...')

    # Train the VAE
    training_loss, validation_loss = train_vae(
        vae, trainer, n_epochs, print_period=1, n_images_train=n_images_train, n_images_valid=n_images_valid
    )
    print('vae trained...')

    plot_loss(n_epochs, training_loss, validation_loss)
    # print('loss plotted...')

    # compare_io(images, outputs)
    # print('comparison done.')


orig_shape = (size, size, 3)
tens_shape = (3, size, size)
if __name__ == '__main__':
    run(n_images_train=250, n_images_valid=100, n_epochs=4)
