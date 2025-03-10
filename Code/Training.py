import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
from Models import create_model_S4D
from DatasetsClass import DataGeneratorPickles
import numpy as np
import random
from Metrics import ESR, STFT_t, STFT_f, RMSE, flux
import sys
import time

def train(**kwargs):
    """
      :param data_dir: the directory in which dataset are stored [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [int]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param model_name: type of the model [string]
      :param dataset: name of the dataset to use [string]
      :param epochs: the number of epochs [int]
      :param order: the order for the Film transformation [int]
      :param act: the activation function [string]
      :param conditioning: if conditioning [bool]
      :param glu: if include GLU [bool]
      :param gcu: if include GCU  [bool]
      :param gaf: if include GAF [bool]

    """
    data_dir = kwargs.get('data_dir', '../../../Files/')
    batch_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-1)
    units = kwargs.get('units', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', None)
    model_name = kwargs.get('model', None)
    epochs = kwargs.get('epochs', 60)
    
    order = kwargs.get('order', 1)
    act = kwargs.get('act', None)
    film = kwargs.get('film', True)
    conditioning = kwargs.get('conditioning', True)
    glu = kwargs.get('glu', True)
    gcu = kwargs.get('gcu', False)
    gaf = kwargs.get('gaf', True)


    start = time.time()
    
    # set all the seed in case reproducibility is desired
    #np.random.seed(422)
    #tf.random.set_seed(422)
    #random.seed(422)

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)

    fs = 48000

    D = 2
    e=d=32
    w = 64
    
    # create the model
    if model_name == 'S4D':
        model = create_model_S4D(D=D, T=w, units=units, order=order, film=film, glu=glu, gcu=gcu, gaf=gaf, act=act, batch_size=batch_size)
    else:
        model = None
   

    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)
    
    # if inference is True, it jump directly to the inference section without train the model
    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        # load the weights of the last epoch, if any
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")

        # create the DataGenerator object to retrieve the data
        train_gen = DataGeneratorPickles(data_dir, dataset + '_train.pickle', input_enc_size=e,
                                          input_dec_size=d, cond_size=D, model=model_name, conditioning=conditioning, batch_size=batch_size)
        val_gen = DataGeneratorPickles(data_dir, dataset + '_val.pickle', input_enc_size=e,
                                        input_dec_size=d, cond_size=D, model=model_name, conditioning=conditioning, batch_size=batch_size)
         
        # the number of total training steps
        training_steps = train_gen.training_steps
        # define the Adam optimizer with the initial learning rate, training steps
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)     
        
        # compile the model with the optimizer and selected loss function
        model.compile(loss='mse', optimizer=opt)

        # defining the array taking the training and validation losses
        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)

        # train the model
        for i in range(epochs):
            print('epochs:', i+1)
            # reset the model's states
            model.reset_states()
            #if model_name == 'S4D':
                #model.layers[3].reset_states()
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=val_gen, callbacks=callbacks)
            
            # store the training and validation loss
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            
        # write and save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs)

        # plot the training and validation loss for all the training
        loss_training = np.array(loss_training[:i])
        loss_val = np.array(loss_val[:i])
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

        print("Training done")
        
        
    avg_time_epoch = (time.time() - start)

    sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch/60)} min")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found, there is something wrong
        print("Something is wrong.")

    # compute test loss
    test_gen = DataGeneratorPickles(data_dir, dataset + '_val.pickle', input_enc_size=e, input_dec_size=d,
                                    cond_size=D, model=model_name, conditioning=conditioning, batch_size=batch_size)
    model.reset_states()
    predictions = model.predict(test_gen, verbose=0)[:, 0]

    # plot and render the output audio file, together with the input and target
    predictWaves(predictions, test_gen.x[w:len(predictions)+w], test_gen.y[w:len(predictions)+w], model_save_dir, save_folder, fs, '0')
     
    mse = tf.keras.metrics.mean_squared_error(test_gen.y[w:len(predictions)+w], predictions)
    mae = tf.keras.metrics.mean_absolute_error(test_gen.y[w:len(predictions)+w], predictions)
    esr = ESR(test_gen.y[w:len(predictions)+w], predictions)
    rmse = RMSE(test_gen.y[w:len(predictions)+w], predictions)
    sftf_t =STFT_t(test_gen.y[w:len(predictions)+w], predictions)
    sftf_f = STFT_f(test_gen.y[w:len(predictions)+w], predictions)
    spectral_flux = flux(test_gen.y[w:len(predictions)+w], predictions, fs)

    results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse, 'spectral_flux': spectral_flux, 'sftf_t': sftf_t, 'sftf_f': sftf_f}

    # write and store the metrics values
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, str(model_name) + 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)
            
    return 42
