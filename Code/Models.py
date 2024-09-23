import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Add, LSTM, Multiply

from Layers import GLU, S4D, GAF, GCU


def create_model_S4D(D, T, units, order, film, glu, gcu, gaf, act=None, b_size=2399, drop=0.):
    """ 
    S4D model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    :param order: order of transformation in the FiLM layer
    :param glu: if true GLU is placed after FiLM
    :param gcu: if true GCU is placed after FiLM
    :param gaf: if true GAF is placed after FiLM
    :param act: activation function
    """
    if film == False and gaf == False:
        T = T + 2
    # Defining decoder inputs
    decoder_inputs = tf.keras.layers.Input(batch_shape=(b_size, T), name='dec_input')
    decoder_outputs = tf.keras.layers.Dense(units//2, input_shape=(b_size, T), name='LinearProjection')(decoder_inputs)
    
    #decoder_outputs = S4D(units//2, b_size)(decoder_outputs)
    #decoder_outputs = tf.keras.layers.Dense(units//2, activation='softsign', name='NonlinearDenseLayer')(decoder_outputs)
    
    #if film:
    #    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')
    #    coeffs = Dense(2*(units // 2), activation=act, batch_input_shape=(b_size, D))(cond_inputs)
    #    g, b = tf.split(coeffs, 2, axis=-1)
    #    if order == 1:
    #        decoder_outputs = Multiply()([decoder_outputs, g])
    #        decoder_outputs = Add()([decoder_outputs, b])

    #    elif order == 3:
    #        decoder_outputs = tf.math.pow(decoder_outputs, 3)
    #        decoder_outputs = Multiply()([decoder_outputs, g])
    #        decoder_outputs = Add()([decoder_outputs, b])
    #    if glu:
    #        decoder_outputs = GLU(in_size=units//2)(decoder_outputs)
    #elif gaf:
    #        cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')
    #        decoder_outputs = GAF(in_size=units//2)(decoder_outputs, cond_inputs)
        
    decoder_outputs = S4D(units//2, b_size)(decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(units//2, activation='softsign', name='NonlinearDenseLayer')(decoder_outputs)
           
    if film:
        cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')
        coeffs = Dense(2*(units // 2), activation=act, batch_input_shape=(b_size, D))(cond_inputs)
        g, b = tf.split(coeffs, 2, axis=-1)
        
        decoder_outputs = tf.math.pow(decoder_outputs, order)
        decoder_outputs = Multiply()([decoder_outputs, g])
        decoder_outputs = Add()([decoder_outputs, b])

        if glu:
            decoder_outputs = GLU(in_size=units//2)(decoder_outputs)
        elif gcu:
            decoder_outputs = GCU(in_size=units//2)(decoder_outputs)

    elif gaf:
            cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')
            decoder_outputs = GAF(in_size=units//2)(decoder_outputs, cond_inputs) 
     
    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    if film or gaf:
        model = tf.keras.models.Model([cond_inputs, decoder_inputs], decoder_outputs)
    else:
        model = tf.keras.models.Model(decoder_inputs, decoder_outputs)

    model.summary()
    return model
