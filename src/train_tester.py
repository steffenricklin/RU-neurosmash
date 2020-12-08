import vae.vae_online_trainer_with_background_removal as bgt
import vae.convvae as conv
batch_size = 10

model = conv.ConvVae(batch_size = batch_size)

background_trainer = bgt.Background_Trainer()
background_trainer.train(model,n_epochs=10, starting_rounds=3,batch_size=batch_size,plot_every= 10)


