import vae.vae_online_trainer_with_background_removal as bgt
from classifier.classifier_test import Classifier_Trainer
from settings import *
from classifier.agent_location_classifier import *

batch_size = 10
model = Agent_Location_Classifier()

background_trainer = Classifier_Trainer()
background_trainer.train(model, n_epochs=20, starting_rounds=5)
background_trainer.test(model,rounds=3)

params_file = f"{param_dir}/vae_1.mx"
model.save_parameters(params_file)

