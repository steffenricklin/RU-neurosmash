import Neurosmash
import utils.background_extractor as BE
from classifier.agent_location_classifier import *
from classifier.agent_locator import *
from utils.converters import *
from mxnet import nd, autograd, gluon, init
import matplotlib.pyplot as plt
import gc


class Classifier_Trainer:
    def __init__(self, env, args):
        self.args = args
        self.agent = Neurosmash.Agent()
        self.env = env

        # extract background
        self.extr = BE.Background_Extractor(self.env, self.agent, args)
        self.background = self.extr.get_background(oned=True)
        self.background_im = self.extr.get_background(oned=False)

    def train(self, model):
        buffer = self.get_initial_buffer(self.args.vision_init_rounds)
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': self.args.vision_lr})
        i = 0
        # Do some loops
        for e in range(self.args.vision_epochs):
            print(f"epoch {e}/{self.args.vision_epochs}")
            epoch_loss = 0
            for im in buffer:
                # reshaped_state = nd.reshape(im, (3, size, size))
                tensor = nd.array(nd.reshape(im, (1, 3, self.args.size, self.args.size)))
                target = locate_agents(im, self.args)
                with autograd.record():
                    out = model(tensor)
                    loss = nd.sum(nd.square(out - target))
                loss.backward()
                trainer.step(1)  # batch size = 1
                epoch_loss += loss
                i += 1
                if i % 50 == 0:
                    image = nd.array(nd.reshape(im, (self.args.size, self.args.size, 3)))
                    self.plot_predictions(image, out)
            buffer = self.update_buffer(buffer)
            gc.collect()

    def test(self, model, rounds):
        data = self.get_single_buffer(rounds)[:rounds]

        for im in data:
            tensor = nd.array(nd.reshape(im, (1, 3, self.args.size, self.args.size)))
            out = model(tensor)
            image = nd.array(nd.reshape(im, (self.args.size, self.args.size, 3)))
            self.plot_predictions(image, out)

    def plot_predictions(self, image, out):
        result_im = image
        outn = out.asnumpy()[0]
        rx, ry, bx, by = map(lambda x: int(x), outn * self.args.size)
        result_im[rx, ry] = [1, 0, 0]
        result_im[bx, by] = [0, 0, 1]
        plt.imshow(result_im.asnumpy())
        plt.show()

    def get_single_buffer(self, max_images=500):
        print("Collecting images")
        buffer = nd.zeros((max_images, state_dim))
        end, reward, state = self.env.reset()
        buffered_images = 0
        while end == 0:
            action = self.agent.step(end, reward, state)
            end, reward, state = self.env.step(action)
            cleanstate = self.extr.clean_and_reshape(state, oned=True)/ 255
            buffer[buffered_images] = cleanstate
            buffered_images += 1
            if buffered_images >= max_images:
                break
        return buffer[:buffered_images]

    def get_initial_buffer(self, n_init_rounds):
        """
        Returns a bunch of states with the bakground removed
        :param n_init_rounds:
        :return:
        """
        print("Collecting images")
        max_images = 500 * n_init_rounds
        buffer = nd.zeros((max_images, state_dim))
        end, reward, state = self.env.reset()
        buffered_images = 0
        rounds = 0
        while buffered_images < max_images and rounds < n_init_rounds:
            end, reward, state = self.env.reset()
            while end == 0:
                action = self.agent.step(end, reward, state)
                end, reward, state = self.env.step(action)
                cleanstate = self.extr.clean_and_reshape(state, oned=True)/ 255
                buffer[buffered_images] = cleanstate
                buffered_images += 1
                if buffered_images >= max_images:
                    break
            rounds += 1
        buffer = buffer[np.random.permutation(buffered_images)]
        return buffer

    def update_buffer(self, buffer):
        # get new data
        new_data = self.get_single_buffer()
        # replace old data with new data
        buffer[:len(new_data)] = new_data
        # shuffle buffer

        new_buffer = buffer[np.random.permutation(len(buffer))]
        del buffer
        del new_data
        return new_buffer

    def plot_in_and_out(self, batch, out):
        fig, ax = plt.subplots(1, 2)
        inp = batch[0].asnumpy().reshape((self.args.size, self.args.size, 3))
        outp = out[0].asnumpy().reshape((self.args.size, self.args.size, 3))
        ax[0].imshow(inp)
        ax[1].imshow(outp)
        plt.show()
        plt.close()
