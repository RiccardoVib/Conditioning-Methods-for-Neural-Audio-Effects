# Copyright (C) 2024 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2024, "Conditioning Methods for Neural Audio Effects" in proceedings of Sound and Music Computing, Porto, Portugal.

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Multiply
from Layers import GLU, GAF, GCU
from S4D import S4D

def create_model_S4D(D, T, units, order, technique, act=None, mini_batch_size=2400, batch_size=1):
    """ 
    S4D model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    :param order: order of transformation in the FiLM layer
    :param technique: conditioning technique
    :param batch_size: batch size
    """
    if technique == 'ExtraInp':
        T = T + 2
    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, T), name='input')
    outputs = tf.keras.layers.Dense(units//2, input_shape=(batch_size, T), name='LinearProjection')(inputs)

    outputs = S4D(model_states=units//2, model_input_dims=units//2, mini_batch_size=mini_batch_size, batch_size=batch_size, stateful=False)(outputs)
    outputs = tf.keras.layers.Dense(units//2, activation='softsign', name='NonlinearDenseLayer')(outputs)

    if technique == 'ExtraInp':
        outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
        model = tf.keras.models.Model(inputs, outputs)
        model.summary()
        return model

    elif technique in ['GAF']:
        cond_inputs = tf.keras.layers.Input(batch_shape=(batch_size, D), name='cond')
        outputs = GAF(in_size=units // 2)(outputs, cond_inputs)
        outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
        model = tf.keras.models.Model([cond_inputs, inputs], outputs)
        model.summary()
        return model

    else:
        cond_inputs = tf.keras.layers.Input(batch_shape=(batch_size, D), name='cond')
        coeffs = Dense(2*(units // 2), activation=act, batch_input_shape=(batch_size, D))(cond_inputs)
        g, b = tf.split(coeffs, 2, axis=-1)

        outputs = tf.math.pow(outputs, order)
        outputs = Multiply()([outputs, g])
        outputs = Add()([outputs, b])

        if technique in ['FILM-GLU']:
            outputs = GLU(in_size=units//2)(outputs)
        elif technique in ['FILM-GCU']:
            outputs = GCU(in_size=units//2)(outputs)

        outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)

        model = tf.keras.models.Model([cond_inputs, inputs], outputs)
        model.summary()
        return model
