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


class GAF(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        """
        Gated Activation function
          :param in_size: input size
          :param bias: if use bias 
          :param dim: dimension for the split
        """
        super(GAF, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.conv_filter = tf.keras.layers.Conv1D(self.in_size*2, kernel_size=self.in_size, strides=1, padding='VALID')
        self.conv_gate = tf.keras.layers.Conv1D(self.in_size*2, kernel_size=2, strides=1, padding='VALID')

    def call(self, x, c):
        x = tf.expand_dims(x, axis=-1)
        c = tf.expand_dims(c, axis=-1)

        cfilter = self.conv_filter(x)
        cgate = self.conv_gate(c)
        out = tf.tanh(cfilter) * tf.sigmoid(cgate)
        return tf.squeeze(out)

class GCU(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        """
        Gated Convolutional Unit
            :param in_size: input size
            :param bias: if use bias 
            :param dim: dimension for the split
        """
        super(GCU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.conv_filter = tf.keras.layers.Conv1D(self.in_size, kernel_size=self.in_size, strides=1, padding='VALID')
        self.conv_gate = tf.keras.layers.Conv1D(self.in_size, kernel_size=self.in_size, strides=1, padding='VALID')

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        out = self.conv_filter(x)
        gate = self.conv_gate(x)
        out = tf.keras.activations.softsign(out)
        out = tf.multiply(out, gate)
        return tf.squeeze(out)    
    
class GLU(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        """
        Gated Linear Unit
            :param in_size: input size
            :param bias: if use bias 
            :param dim: dimension for the split
        """
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size*2, use_bias=bias)

    def call(self, x):
        x = self.dense(x)
        out, gate = tf.split(x, 2, axis=self.dim)
        gate = tf.keras.activations.softsign(gate)
        x = tf.multiply(out, gate)
        return x