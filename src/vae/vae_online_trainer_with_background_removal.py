import Neurosmash
from settings import *
import numpy as np
import utils.background_extractor as BE
from settings import *
from mxnet import nd, autograd, gluon
import mxnet as mx
import matplotlib.pyplot as plt

class Background_Trainer:

    def __init__(self, env):
        self.agent = Neurosmash.Agent()
        self.env = env
        extr = BE.Background_Extractor(self.env, self.agent)
        self.background = extr.get_background().reshape(-1)

    def train(self, model, n_epochs, starting_rounds, batch_size = 1, plot_every=10):
        buffer = self.get_initial_buffer(starting_rounds)
        model.collect_params().initialize(mx.init.Xavier(), ctx=model.ctx)
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': .0001})
        i = 0
        for e in range(n_epochs):
            print(f"Epoch {e}/{n_epochs}")
            n_batches = int(len(buffer) / batch_size)
            for b in range(n_batches):
                print(f"Batch {b}/{n_batches}")
                batch = buffer[b*batch_size: (b+1)*batch_size]
                batch_reshaped = nd.reshape(batch, (batch_size, 3, size, size))
                with autograd.record():
                    out, loss = model(batch_reshaped)
                loss.backward()
                trainer.step(batch_size)  # batch size = 1
                i+=1
                # if i%plot_every == 0:
                #     self.plot_in_and_out(batch, out)
            buffer = self.update_buffer(buffer)


    def plot_in_and_out(self, batch, out):
        fig, ax = plt.subplots(1, 2)
        inp = batch[0].asnumpy().reshape((size, size, 3))
        outp = out[0].asnumpy().reshape((size, size, 3))
        ax[0].imshow(inp)
        ax[1].imshow(outp)
        plt.show()
        plt.close()

    def update_buffer(self, buffer):
        # get new data
        new_data = self.get_single_buffer()
        # replace old data with new data
        buffer[:len(new_data)] = new_data
        # shuffle buffer
        return buffer[np.random.permutation(len(buffer))]

    def get_initial_buffer(self, n_init_rounds):
        """
        Returns a bunch of states with the bakground removed
        :param n_init_rounds:
        :return:
        """
        max_images = n_init_rounds*500
        buffer = nd.zeros((max_images, state_dim))
        end, reward, state = self.env.reset()
        buffered_images = 0
        rounds = 0
        while buffered_images < max_images and rounds < n_init_rounds:
            end, reward, state = self.env.reset()
            while end == 0:
                action = self.agent.step(end, reward, state)
                end, reward, state = self.env.step(action)
                cleanstate = np.where(state == self.background, 200, state)
                buffer[buffered_images] = cleanstate/255
                buffered_images += 1
                if buffered_images >= max_images:
                    break
            rounds += 1
        buffer = buffer[np.random.permutation(buffered_images)]
        return buffer

    def get_single_buffer(self):
        max_images = 500
        buffer = nd.zeros((max_images, state_dim))
        end, reward, state = self.env.reset()
        buffered_images = 0
        while end == 0:
            action = self.agent.step(end, reward, state)
            end, reward, state = self.env.step(action)
            cleanstate = np.where(state == self.background, 200, state)
            buffer[buffered_images] = cleanstate/255
            buffered_images += 1
            if buffered_images >= max_images:
                break
        return buffer[:buffered_images]