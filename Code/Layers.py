import tensorflow as tf
import math
import numpy as np
from einops import repeat

class GAF(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
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
        super(GCU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.conv_filter = tf.keras.layers.Conv1D(self.in_size, kernel_size=self.in_size, strides=1, padding='VALID')
        self.conv_gate = tf.keras.layers.Conv1D(self.in_size, kernel_size=self.in_size, strides=1, padding='VALID')

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        #c = tf.expand_dims(c, axis=-1)
        out = self.conv_filter(x)
        gate = self.conv_gate(x)
        out = tf.keras.activations.softsign(out)
        out = tf.multiply(out, gate)
        return tf.squeeze(out)    
    
class GLU(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
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

### from https://github.com/state-spaces/s4/blob/main/models/s4/
        
class S4DKernel(tf.keras.layers.Layer):
    def __init__(self,  N=64, d_model=1, dt_min=0.001, dt_max=0.1, b_size=600):
        super(S4DKernel, self).__init__()
        self.N = N
        # Generate dt
        self.H = d_model
        self.log_dt = tf.random.uniform((self.H,), minval=tf.math.log(dt_min), maxval=tf.math.log(dt_max))

        #self.B = tf.Variable(0.5 * tf.ones((self.H, self.N//2)), trainable=True, dtype='float32')

        C = tf.complex(tf.random.normal([self.H, self.N // 2]), tf.random.normal([self.H, self.N // 2]))
        self.C = tf.Variable(tf.cast(C, dtype=tf.float32), trainable=True)

        self.log_dt = tf.Variable(self.log_dt, trainable=True)

        log_A_real = tf.math.log(tf.constant(0.5 * tf.ones((self.H, self.N//2))))
        A_imag = tf.constant(np.pi) * repeat(np.arange(N//2), 'n -> h n', h=self.H)

        self.log_A_real = tf.Variable(log_A_real, trainable=True)
        self.A_imag = tf.Variable(A_imag, trainable=True)

        self.b_size = b_size
        self.reset_states()

    def call(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = tf.exp(self.log_dt)  # (H)
        C = tf.cast(self.C, dtype=tf.complex64)  # (H N)

        A = tf.dtypes.complex(-tf.exp(self.log_A_real), self.A_imag)

        # Vandermonde multiplication
        dt = tf.expand_dims(dt, axis=-1)
        dtA = tf.cast(tf.math.real(A) * dt, dtype=tf.complex64)   # (H N)

        ####States
        # Augment B with state
        # s = tf.math.real(self.state) / dt
        # s = s * dtA * tf.exp(dtA) / (tf.exp(dtA) - 1.)
        # B = tf.concat([s, self.B], axis=-3)  # (1+B H N)
        # # Combine B and C
        # C = tf.reshape(B[:, None, :, :] * C, [-1, self.H, self.N])

        K = tf.expand_dims(dtA, axis=-1) * tf.cast(tf.range(L), dtype=tf.complex64)  # (H N L)
        C = C * (tf.exp(dtA) - 1.) / A
        K = tf.math.real(2 * tf.einsum('hn, hnl -> hl', C, tf.exp(K)))
        #K = 2 * tf.reduce_sum(C * tf.exp(K), axis=1, keepdims=False)

        ####States
        # K = tf.reshape(K, [-1, 1, self.H, L])  # (1+B C H L)
        # self.state = K[:-1, :, :, :]  # (B C H L)
        # K = K[-1, :, :, :]  # (C H L)

        return K

    def reset_states(self):
        self.state = tf.zeros((self.H, self.N//2), dtype=tf.complex64)

class S4D(tf.keras.layers.Layer):
    def __init__(self, d_state=64, d_model=1, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = tf.Variable(tf.random.normal([self.h]), dtype='float32')

        # SSM Kernel
        self.kernel = S4DKernel(N=self.n, d_model=self.h, **kernel_args)

        # Pointwise
        self.activation = tf.keras.activations.gelu
        self.glu = GLU(in_size=2*self.n, dim=-1)

    def call(self, u):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        L = u.shape[-1]

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)
        # Convolution
        k_f = tf.signal.rfft(k, fft_length=[2 * L])  # (H L)
        u_f = tf.signal.rfft(u, fft_length=[2 * L])  # (B H L)
        y = tf.signal.irfft((u_f * k_f), fft_length=[2 * L])[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * tf.expand_dims(self.D, axis=-1)
        y = self.glu(y)

        return y
