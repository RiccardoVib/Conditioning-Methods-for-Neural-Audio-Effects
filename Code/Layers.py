import tensorflow as tf
import math
import tensorflow_probability as tfp
import numpy as np

parallel_scan = tfp.math.scan_associative
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


class S4DKernel(tf.keras.layers.Layer):
    """Generate convolution kernel from diagonal SSM parameters.
        A: (S, N) diagonal matrix
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, b_size=600):
        super().__init__()
        self.N = N
        # Generate dt
        self.H = d_model
        self.log_dt = tf.random.uniform([self.H]) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        
        self.log_dt = tf.Variable(self.log_dt, trainable=True)

        self.B = tf.Variable(0.5 * tf.ones((self.H, self.N//2)), trainable=True)
        #B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()


        C = tf.random.normal([self.H, self.N // 2]) + 1j * tf.random.normal([self.H, self.N // 2]).numpy()
        self.C = tf.Variable(tf.math.real(C), trainable=True)
        self.log_dt = tf.Variable(self.log_dt)

        self.log_A_real = tf.Variable(tf.math.log(0.5 * tf.ones((self.H, self.N//2))), trainable=True)
        self.A_imag = tf.Variable(math.pi * repeat(np.arange(N//2), 'n -> h n', h=self.H), trainable=True)

        self.b_size=b_size
        self.state = self.reset_states()

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = tf.exp(self.log_dt)  # (H)
        C = tf.cast(self.C, dtype=tf.complex64)  # (H N)
        A = -tf.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)

        ####States
        # Augment B with state
        s = self.state / dt
        s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
        B = tf.concat([s, self.B], axis=-3)  # (1+B H N)
        # Combine B and C
        C = tf.reshape(B[:, None, :, :] * C, [-1, self.H, self.N])


        K = dtA.unsqueeze(-1) * tf.arange(L, device=A.device)  # (H N L)
        C = C * (tf.exp(dtA) - 1.) / A
        K = 2 * tf.einsum('hn, hnl -> hl', C, tf.exp(K)).real

        ####States
        K = tf.reshape(K, [-1, 1, self.H, L])  # (1+B C H L)
        self.state = K[:-1, :, :, :]  # (B C H L)
        K = K[-1, :, :, :]  # (C H L)

        return K

    def reset_states(self):
        self.state = tf.zeros((self.b_size, self.H, self.N), dtype=tf.complex64)

class S4D(tf.keras.layers.Layer):
    def __init__(self, d_model, d_state=64, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = tf.Variable(tf.random.normal([self.h]))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = tf.keras.activations.gelu

        # position-wise output transform to mix features
        # self.output_linear = tf.keras.Sequential(
        #     Conv1D(self.h, 2*self.h, kernel_size=1),
        #     tf.nn.GLU(dim=-2),
        # )
        self.output_linear = tf.keras.layers.Conv1D(2 * self.h, kernel_size=1)
        self.output_linear2 = GLU(in_size=2*self.h, dim=-2)

    def forward(self, u):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = tf.fft.rfft(k, n=2 * L)  # (H L)
        u_f = tf.fft.rfft(u, n=2 * L)  # (B H L)
        y = tf.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        y = self.output_linear2(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None  # Return a dummy state to satisfy this repo's interface, but this can be modified


