"""
    Implementation of the compelx memory cells of our NIPS-Paper.
    Including:
        1.) The original URNN-cell.
        2.) Our Phase-Relu cell.
"""
import collections
import numpy as np
import tensorflow as tf
from tensorflow import random_uniform_initializer as urnd_init
from src.custom_regularizers import complex_dropout
from tensorflow.contrib.rnn import LSTMStateTuple


def hilbert(xr):
    '''
    Implements the hilbert transform, a mapping from C to R.
    Args:
        xr: The input sequence.
    Returns:
        xc: A complex sequence of the same length.
    '''
    with tf.variable_scope('hilbert_transform'):
        n = tf.Tensor.get_shape(xr).as_list()[0]
        # Run the fft on the columns no the rows.
        x = tf.transpose(tf.fft(tf.transpose(xr)))
        h = np.zeros([n])
        if n > 0 and 2*np.fix(n/2) == n:
            # even and nonempty
            h[0:int(n/2+1)] = 1
            h[1:int(n/2)] = 2
        elif n > 0:
            # odd and nonempty
            h[0] = 1
            h[1:int((n+1)/2)] = 2
        tf_h = tf.constant(h, name='h', dtype=tf.float32)
        if len(x.shape) == 2:
            hs = np.stack([h]*x.shape[-1], -1)
            reps = tf.Tensor.get_shape(x).as_list()[-1]
            hs = tf.stack([tf_h]*reps, -1)
        elif len(x.shape) == 1:
            hs = tf_h
        else:
            raise NotImplementedError
        tf_hc = tf.complex(hs, tf.zeros_like(hs))
        xc = x*tf_hc
        return tf.transpose(tf.ifft(tf.transpose(xc)))


def unitary_init(shape, dtype=tf.float32, partition_info=None):
    '''
    Initialize using an unitary matrix, generated by using an SVD and
    multiplying UV, while discarding the signular value matrix.
    '''
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    rand_r = np.random.uniform(-limit, limit, shape[0:2])
    rand_i = np.random.uniform(-limit, limit, shape[0:2])
    crand = rand_r + 1j*rand_i
    u, s, vh = np.linalg.svd(crand)
    # use u and vg to create a unitary matrix:
    unitary = np.matmul(u, np.transpose(np.conj(vh)))

    test_eye = np.matmul(np.transpose(np.conj(unitary)), unitary)
    print('I - Wi.H Wi', np.linalg.norm(test_eye) - unitary)
    # test
    # plt.imshow(np.abs(np.matmul(unitary, np.transpose(np.conj(unitary))))); plt.show()
    stacked = np.stack([np.real(unitary), np.imag(unitary)], -1)
    assert stacked.shape == tuple(shape), "Unitary initialization shape mismatch."
    # debug_here()
    return tf.constant(stacked, dtype)


def arjovski_init(shape, dtype=tf.float32, partition_info=None):
    '''
    Use Arjovsky's unitary basis as initalization.
    Reference:
         Arjovsky et al. Unitary Evolution Recurrent Neural Networks
         https://arxiv.org/abs/1511.06464
    '''
    print("Arjosky basis initialization.")
    assert shape[0] == shape[1]
    omega1 = np.random.uniform(-np.pi, np.pi, shape[0])
    omega2 = np.random.uniform(-np.pi, np.pi, shape[0])
    omega3 = np.random.uniform(-np.pi, np.pi, shape[0])

    vr1 = np.random.uniform(-1, 1, [shape[0], 1])
    vi1 = np.random.uniform(-1, 1, [shape[0], 1])
    v1 = vr1 + 1j*vi1
    vr2 = np.random.uniform(-1, 1, [shape[0], 1])
    vi2 = np.random.uniform(-1, 1, [shape[0], 1])
    v2 = vr2 + 1j*vi2

    D1 = np.diag(np.exp(1j*omega1))
    D2 = np.diag(np.exp(1j*omega2))
    D3 = np.diag(np.exp(1j*omega3))

    vvh1 = np.matmul(v1, np.transpose(np.conj(v1)))
    beta1 = 2./np.matmul(np.transpose(np.conj(v1)), v1)
    R1 = np.eye(shape[0]) - beta1*vvh1

    vvh2 = np.matmul(v2, np.transpose(np.conj(v2)))
    beta2 = 2./np.matmul(np.transpose(np.conj(v2)), v2)
    R2 = np.eye(shape[0]) - beta2*vvh2

    perm = np.random.permutation(np.eye(shape[0], dtype=np.float32)) \
        + 1j*np.zeros(shape[0])

    fft = np.fft.fft
    ifft = np.fft.ifft

    step1 = fft(D1)
    step2 = np.matmul(R1, step1)
    step3 = np.matmul(perm, step2)
    step4 = np.matmul(D2, step3)
    step5 = ifft(step4)
    step6 = np.matmul(R2, step5)
    unitary = np.matmul(D3, step6)
    eye_test = np.matmul(np.transpose(np.conj(unitary)), unitary)
    unitary_test = np.linalg.norm(np.eye(shape[0]) - eye_test)
    print('I - Wi.H Wi', unitary_test, unitary.dtype)
    assert unitary_test < 1e-10, "Unitary initialization not unitary enough."
    stacked = np.stack([np.real(unitary), np.imag(unitary)], -1)
    assert stacked.shape == tuple(shape), "Unitary initialization shape mismatch."
    return tf.constant(stacked, dtype)


def mod_relu(z, scope='', reuse=None):
    """
        Implementation of the modRelu from Arjovski et al.
        f(z) = relu(|z| + b)(z / |z|) or
        f(r,theta) = relu(r + b)e^(i*theta)
        b is initialized to zero, this leads to a network, which
        is linear during early optimization.
    Input:
        z: complex input.
        b: 'dead' zone radius.
    Returns:
        z_out: complex output.

    Reference:
         Arjovsky et al. Unitary Evolution Recurrent Neural Networks
         https://arxiv.org/abs/1511.06464
    """
    with tf.variable_scope('mod_relu' + scope, reuse=reuse):
        b = tf.get_variable('b', [], dtype=tf.float32,
                            initializer=urnd_init(-0.01, 0.01))
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        rescale = tf.nn.relu(modulus + b) / (modulus + 1e-6)
        # return tf.complex(rescale * tf.real(z),
        #                   rescale * tf.imag(z))
        rescale = tf.complex(rescale, tf.zeros_like(rescale))
        return tf.multiply(rescale, z)


def relu(x, scope='', reuse=None):
    return tf.nn.relu(x)


def tanh(x, scope='', reuse=None):
    return tf.nn.tanh(x)


def split_relu(z, scope='', reuse=None):
    with tf.variable_scope('split_relu' + scope):
        x = tf.real(z)
        y = tf.imag(z)
        return tf.complex(tf.nn.relu(x), tf.nn.relu(y))


def z_relu(z, scope='', reuse=None):
    with tf.variable_scope('z_relu'):
        factor1 = tf.cast(tf.real(z) > 0, tf.float32)
        factor2 = tf.cast(tf.imag(z) > 0, tf.float32)
        combined = factor1*factor2
        rescale = tf.complex(combined, tf.zeros_like(combined))
        return tf.multiply(rescale, z)


def hirose(z, scope='', reuse=None):
    """
    Compute the non-linearity proposed by Hirose.
    """
    with tf.variable_scope('hirose' + scope, reuse=reuse):
        m = tf.get_variable('m', [], tf.float32,
                            initializer=urnd_init(0.9, 1.1))
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        # use m*m to enforce positive m.
        rescale = tf.complex(tf.nn.tanh(modulus/(m*m))/modulus,
                             tf.zeros_like(modulus))
        return tf.multiply(rescale, z)


def double_sigmoid(z, scope='', reuse=None):
    """
    ModSigmoid implementation, using a coupled alpha and beta.
    """
    with tf.variable_scope('double_sigmoid' + scope, reuse=reuse):
        return tf.complex(tf.nn.sigmoid(tf.real(z)),
                          tf.nn.sigmoid(tf.imag(z)))


def single_sigmoid_real(z, scope='', reuse=None):
    """
    ModSigmoid implementation, using a coupled alpha and beta.
    """
    with tf.variable_scope('sigmoid_real_' + scope, reuse=reuse):
        rz = tf.nn.sigmoid(tf.real(z))
        return tf.complex(rz, tf.zeros_like(rz))


def single_sigmoid_imag(z, scope='', reuse=None):
    """
    ModSigmoid implementation, using a coupled alpha and beta.
    """
    with tf.variable_scope('sigmoid_imag_' + scope, reuse=reuse):
        iz = tf.nn.sigmoid(tf.imag(z))
        return tf.complex(iz, tf.zeros_like(iz))


def mod_sigmoid(z, scope='', reuse=None):
    """
    ModSigmoid implementation, using a coupled alpha and beta.
    """
    with tf.variable_scope('mod_sigmoid_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        alpha_norm = tf.nn.sigmoid(alpha)
        pre_act = alpha_norm * tf.real(z) + (1 - alpha_norm)*tf.imag(z)
        return tf.complex(tf.nn.sigmoid(pre_act), tf.zeros_like(pre_act))


def mod_sigmoid_beta(z, scope='', reuse=None):
    """
    ModSigmoid implementation, with uncoupled alpha and beta.
    """
    with tf.variable_scope('mod_sigmoid_beta_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        beta = tf.get_variable('beta', [], dtype=tf.float32,
                               initializer=tf.constant_initializer(1.0))
        alpha_norm = tf.nn.sigmoid(alpha)
        beta_norm = tf.nn.sigmoid(beta)
        pre_act = alpha_norm * tf.real(z) + beta_norm*tf.imag(z)
        return tf.complex(tf.nn.sigmoid(pre_act), tf.zeros_like(pre_act))


def real_mod_sigmoid_beta(z, scope='', reuse=None):
    """
    Real valued ModSigmoid implementation, with uncoupled alpha and beta.
    """
    with tf.variable_scope('real_mod_sigmoid_beta_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        beta = tf.get_variable('beta', [], dtype=tf.float32,
                               initializer=tf.constant_initializer(1.0))
        alpha_norm = tf.nn.sigmoid(alpha)
        beta_norm = tf.nn.sigmoid(beta)
        pre_act = alpha_norm*z[0] + beta_norm*z[1]
        return tf.nn.sigmoid(pre_act)


def mod_sigmoid_gamma(z, scope='', reuse=None):
    """
    ModSigmoid implementation, with uncoupled and unbounded
    alpha and beta.
    """
    with tf.variable_scope('mod_sigmoid_beta_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        beta = tf.get_variable('beta', [], dtype=tf.float32,
                               initializer=tf.constant_initializer(1.0))
        pre_act = alpha * tf.real(z) + beta*tf.imag(z)
        return tf.complex(tf.nn.sigmoid(pre_act), tf.zeros_like(pre_act))


def mod_sigmoid_prod(z, scope='', reuse=None):
    """
    Product version of the mod sigmoid.
    """
    with tf.variable_scope('mod_sigmoid_prod_' + scope, reuse=reuse):
        prod = tf.nn.sigmoid(tf.real(z)) * tf.nn.sigmoid(tf.imag(z))
        return tf.complex(prod, tf.zeros_like(prod))


def mod_sigmoid_sum(z, scope='', reuse=None):
    with tf.variable_scope('mod_sigmoid_sum_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        sig_alpha = tf.nn.sigmoid(alpha)
        sig_sum = (sig_alpha*tf.nn.sigmoid(tf.real(z))
                   + (1.0 - sig_alpha) * tf.nn.sigmoid(tf.imag(z)))
        return tf.complex(sig_sum, tf.zeros_like(sig_sum))


def mod_sigmoid_sum_beta(z, scope='', reuse=None):
    """ Probably not a good idea. """
    with tf.variable_scope('mod_sigmoid_sum_beta_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        beta = tf.get_variable('beta', [], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        sig_alpha = tf.nn.sigmoid(alpha)
        sig_beta = tf.nn.sigmoid(beta)
        sig_sum = (sig_alpha*tf.nn.sigmoid(tf.real(z))
                   + sig_beta * tf.nn.sigmoid(tf.imag(z)))
        return tf.complex(sig_sum, tf.zeros_like(sig_sum))


def mod_sigmoid_split(z, scope='', reuse=None):
    """
    ModSigmoid implementation applying a sigmoid on the imaginary
    and real parts seperately.
    """
    with tf.variable_scope('mod_sigmoid_split_' + scope, reuse=reuse):
        return tf.complex(tf.nn.sigmoid(tf.real(z)), tf.nn.sigmoid(tf.imag(z)))


def gate_phase_hirose(z, scope='', reuse=None):
    '''
    Hirose inspired gate activation filtering according to
    phase angle.
    '''
    with tf.variable_scope('phase_hirose_' + scope, reuse=reuse):
        m = tf.get_variable('m', [], tf.float32,
                            initializer=urnd_init(0.9, 1.1))
        a = tf.get_variable('a', [], tf.float32,
                            initializer=urnd_init(1.9, 2.1))
        b = tf.get_variable('b', [], tf.float32, urnd_init(3.9, 4.1))
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        phase = tf.atan2(tf.imag(z), tf.real(z))
        gate = tf.tanh(modulus/(m*m)) * tf.nn.sigmoid(a*phase + b)
        return tf.complex(gate, tf.zeros_like(gate))


def moebius(z, scope='', reuse=None):
    """
    Implement a learnable moebius transformation.
    """
    with tf.variable_scope('moebius' + scope, reuse=reuse):
        ar = tf.get_variable('ar', [], tf.float32,
                             initializer=tf.constant_initializer(1))
        ai = tf.get_variable('ai', [], tf.float32,
                             initializer=tf.constant_initializer(0))
        b = tf.get_variable('b', [2], tf.float32,
                            initializer=tf.constant_initializer(0))
        c = tf.get_variable('c', [2], tf.float32,
                            initializer=tf.constant_initializer(0))
        dr = tf.get_variable('dr', [], tf.float32,
                             initializer=tf.constant_initializer(1))
        di = tf.get_variable('di', [], tf.float32,
                             initializer=tf.constant_initializer(0))

        a = tf.complex(ar, ai)
        b = tf.complex(b[0], b[1])
        c = tf.complex(c[0], c[1])
        d = tf.complex(dr, di)
        return tf.divide(tf.multiply(a, z) + b,
                         tf.multiply(c, z) + d)


def linear(z, scope='', reuse=None, coupled=False):
    return z


def rfl_mul(h, state_size, no, reuse):
    """
    Multiplication with a reflection.
    Implementing R = I - (vv*/|v|^2)
    Input:
        h: hidden state_vector.
        state_size: The RNN state size.
        reuse: True if graph variables should be reused.
    Returns:
        R*h
    """
    with tf.variable_scope("reflection_v_" + str(no), reuse=reuse):
        vr = tf.get_variable('vr', shape=[state_size, 1], dtype=tf.float32,
                             initializer=tf.glorot_uniform_initializer())
        vi = tf.get_variable('vi', shape=[state_size, 1], dtype=tf.float32,
                             initializer=tf.glorot_uniform_initializer())

    with tf.variable_scope("ref_mul_" + str(no), reuse=reuse):
        hr = tf.real(h)
        hi = tf.imag(h)
        vstarv = tf.reduce_sum(vr**2 + vi**2)
        hr_vr = tf.matmul(hr, vr)
        hr_vi = tf.matmul(hr, vi)
        hi_vr = tf.matmul(hi, vr)
        hi_vi = tf.matmul(hi, vi)

        # tf.matmul with transposition is the same as T.outer
        # we need something of the shape [batch_size, state_size] in the end
        a = tf.matmul(hr_vr - hi_vi, vr, transpose_b=True)
        b = tf.matmul(hr_vi + hi_vr, vi, transpose_b=True)
        c = tf.matmul(hr_vr - hi_vi, vi, transpose_b=True)
        d = tf.matmul(hr_vi + hi_vr, vr, transpose_b=True)

        # the thing we return is:
        # return_re = hr - (2/vstarv)(d - c)
        # return_im = hi - (2/vstarv)(a + b)
        new_hr = hr - (2.0 / vstarv) * (a + b)
        new_hi = hi - (2.0 / vstarv) * (d - c)
        new_state = tf.complex(new_hr, new_hi)

        # v = tf.complex(vr, vi)
        # vstarv = tf.complex(tf.reduce_sum(vr**2 + vi**2), 0.0)
        # # vstarv = tf.matmul(tf.transpose(tf.conj(v)), v)
        # vvstar = tf.matmul(v, tf.transpose(tf.conj(v)))
        # refsub = (2.0/vstarv)*vvstar
        # R = tf.identity(refsub) - refsub
        return new_state


def diag_mul(h, state_size, no, reuse):
    """
    Multiplication with a diagonal matrix.
    Input:
        h: hidden state_vector.
        state_size: The RNN state size.
        reuse: True if graph variables should be reused.
    Returns:
        R*h
    """
    with tf.variable_scope("diag_phis_" + str(no), reuse=reuse):
        omega = tf.get_variable('vr', shape=[state_size], dtype=tf.float32,
                                initializer=urnd_init(-np.pi, np.pi))
        dr = tf.cos(omega)
        di = tf.sin(omega)

    with tf.variable_scope("diag_mul_" + str(no)):
        D = tf.diag(tf.complex(dr, di))
        return tf.matmul(h, D)


def permutation(h, state_size, no, reuse):
    """
    Apply a random permutation to the RNN state.
    Input:
        h: the original state.
    Output:
        hp: the permuted state.
    """
    with tf.variable_scope("permutation_" + str(no), reuse):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            return np.random.permutation(np.eye(state_size, dtype=np.float32))
        Pr = tf.get_variable("Permutation", dtype=tf.float32,
                             initializer=_initializer, shape=[state_size],
                             trainable=False)
        P = tf.complex(Pr, tf.constant(0.0, dtype=tf.float32))
    return tf.matmul(h, P)


def matmul_plus_bias(x, num_proj, scope, reuse, bias=True,
                     bias_init=0.0, orthogonal=False):
    """
    Compute Ax + b.
    Input: x
    Returns: Ax + b
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    with tf.variable_scope(scope, reuse=reuse):
        if orthogonal:
            with tf.variable_scope('orthogonal_stiefel', reuse=reuse):
                A = tf.get_variable('gate_O', [in_shape[-1], num_proj],
                                    dtype=tf.float32,
                                    initializer=tf.orthogonal_initializer())
        else:
            A = tf.get_variable('A', [in_shape[-1], num_proj], dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())
        if bias:
            b = tf.get_variable('bias', [num_proj], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_init))
            print('Initializing', tf.contrib.framework.get_name_scope(), 'bias to',
                  bias_init)
            return tf.matmul(x, A) + b
        else:
            return tf.matmul(x, A)


def complex_matmul(x, num_proj, scope, reuse, bias=False, bias_init_r=0.0,
                   bias_init_c=0.0, unitary=False, orthogonal=False,
                   unitary_init=arjovski_init):
    """
    Compute Ax + b.
    Input: x
    Returns: Ax + b
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    # debug_here()
    with tf.variable_scope(scope, reuse=reuse):
        if unitary:
            with tf.variable_scope('unitary_stiefel', reuse=reuse):
                varU = tf.get_variable('gate_U',
                                       shape=in_shape[-1:] + [num_proj] + [2],
                                       dtype=tf.float32,
                                       initializer=unitary_init)
                A = tf.complex(varU[:, :, 0], varU[:, :, 1])
        elif orthogonal:
            with tf.variable_scope('orthogonal_stiefel', reuse=reuse):
                Ar = tf.get_variable('gate_Ur', in_shape[-1:] + [num_proj],
                                     dtype=tf.float32,
                                     initializer=tf.orthogonal_initializer())
                Ai = tf.get_variable('gate_Ui', in_shape[-1:] + [num_proj],
                                     dtype=tf.float32,
                                     initializer=tf.orthogonal_initializer())
                A = tf.complex(Ar, Ai)
        else:
            varU = tf.get_variable('gate_A',
                                   shape=in_shape[-1:] + [num_proj] + [2],
                                   dtype=tf.float32,
                                   initializer=tf.glorot_uniform_initializer())
            A = tf.complex(varU[:, :, 0], varU[:, :, 1])
        if bias:
            varbr = tf.get_variable('bias_r', [num_proj], dtype=tf.float32,
                                    initializer=tf.constant_initializer(bias_init_r))
            varbc = tf.get_variable('bias_c', [num_proj], dtype=tf.float32,
                                    initializer=tf.constant_initializer(bias_init_c))
            b = tf.complex(varbr, varbc)
            return tf.matmul(x, A) + b
        else:
            return tf.matmul(x, A)


def C_to_R(h, num_proj, reuse, scope=None, bias_init=0.0):
    '''
    Linear mapping from C to R.
    '''
    with tf.variable_scope(scope or "C_to_R"):
        concat = tf.concat([tf.real(h), tf.imag(h)], axis=-1)
        return matmul_plus_bias(concat, num_proj, 'final', reuse, bias_init)


class UnitaryCell(tf.nn.rnn_cell.RNNCell):
    """
    Tensorflow implementation of unitary evolution RNN as proposed by Arjosky et al.
    """

    def __init__(self, num_units, activation=mod_relu, num_proj=None, reuse=None,
                 real=False, complex_input=False):
        super().__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._output_size = num_proj
        self._arjovski_basis = False
        self._real = real
        self._complex_input = complex_input

    def to_string(self):
        cell_str = 'UnitaryCell' + '_' \
            + '_' + 'activation' + '_' + str(self._activation.__name__) + '_' \
            + '_arjovski_basis' + '_' + str(self._arjovski_basis) + '_' \
            + '_real_cell_' + str(self._real)
        return cell_str

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        if self._output_size is None:
            return self._num_units
        else:
            return self._output_size

    def zero_state(self, batch_size, dtype=tf.float32):
        out = tf.zeros([batch_size, self._output_size], dtype=tf.float32)
        if self._real:
            rnd = tf.random_uniform([batch_size, self._num_units],
                                    minval=0, maxval=2)
            first_state = rnd/tf.norm(rnd)
        else:
            omegas = tf.random_uniform([batch_size, self._num_units],
                                       minval=0, maxval=2*np.pi)
            sx = tf.cos(omegas)
            sy = tf.sin(omegas)
            r = (1.0)/np.sqrt(self._num_units)
            first_state = tf.complex(r*sx, r*sy)
        return LSTMStateTuple(out, first_state)

    def call(self, inputs, state):
        """
            Evaluate the RNN cell. Using
            h_(t+1) = U_t*f(h_t) + V_t x_t
        """
        # with tf.variable_scope("UnitaryCell"):
        last_out, last_h = state
        if self._real:
            with tf.variable_scope("orthogonal_stiefel"):
                matO = tf.get_variable("recurrent_O",
                                       shape=[self._num_units, self._num_units],
                                       dtype=tf.float32,
                                       initializer=tf.orthogonal_initializer())
                Uh = tf.matmul(last_h, matO)
        elif self._arjovski_basis:
            with tf.variable_scope("arjovski_basis", reuse=self._reuse):
                step1 = diag_mul(last_h, self._num_units, 0, self._reuse)
                step2 = tf.spectral.fft(step1)
                step3 = rfl_mul(step2, self._num_units, 0, self._reuse)
                step4 = permutation(step3, self._num_units, 0, self._reuse)
                step5 = diag_mul(step4, self._num_units, 1, self._reuse)
                step6 = tf.spectral.ifft(step5)
                step7 = rfl_mul(step6, self._num_units, 1, self._reuse)
                Uh = diag_mul(step7, self._num_units, 2, self._reuse)
        else:
            with tf.variable_scope("unitary_stiefel", reuse=self._reuse):
                varU = tf.get_variable("recurrent_U",
                                       shape=[self._num_units, self._num_units, 2],
                                       dtype=tf.float32,
                                       initializer=arjovski_init)
                U = tf.complex(varU[:, :, 0], varU[:, :, 1])
                # U = tf.Print(U, [U])
                Uh = tf.matmul(last_h, U)

        # Deal with the inputs.
        if self._real:
            Vx = matmul_plus_bias(inputs, self._num_units, 'Vx', self._reuse)
        else:
            if self._complex_input:
                cin = inputs
            else:
                cin = tf.complex(inputs, tf.zeros_like(inputs))
            Vx = complex_matmul(cin, self._num_units, 'Vx', self._reuse, bias=True)

        # By FFT.
        # TODO.

        zt = Uh + Vx
        ht = self._activation(zt, '', self._reuse)

        # Mapping the state back onto the real axis.
        # By mapping.
        if not self._real:
            output = C_to_R(ht, self._output_size, reuse=self._reuse)
        else:
            output = matmul_plus_bias(ht, self._output_size, 'final', self._reuse, 0.0)

        # By fft.
        # TODO.
        newstate = LSTMStateTuple(output, ht)
        return output, newstate


class StiefelGatedRecurrentUnit(tf.nn.rnn_cell.RNNCell):
    '''
    Implementation of a Stiefel Gated Recurrent unit.
    '''

    def __init__(self, num_units, activation=hirose,
                 gate_activation=mod_sigmoid,
                 num_proj=None, reuse=None, stiefel=True,
                 real=False, real_double=False,
                 complex_input=False,
                 complex_output=True,
                 dropout=False):
        """
        Params:
            num_units: The size of the hidden state.
            activation: State to state non-linearity.
            gate_activation: The gating non-linearity.
            num_proj: Output dimension.
            reuse: Reuse graph weights in existing scope.
            stiefel: If True the cell will be used using the Stiefel
                     optimization scheme from wisdom et al.
            real: If true a real valued cell will be created.
            real_double: Use a doulbe real gate similar to to
                         the complex version.
        """
        super().__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        # self._state_to_state_act = linear
        self._output_size = num_proj
        self._arjovski_basis = False
        self._input_fourier = False
        self._input_hilbert = False
        self._input_split_matmul = False
        self._stiefel = stiefel
        self._gate_activation = gate_activation
        self._single_gate = False
        self._real = real
        self._real_double = False
        self._complex_output = complex_output
        self._complex_input = complex_input
        self._dropout = dropout

    def to_string(self):
        cell_str = 'cGRU' + '_' \
            + '_' + 'act' + '_' + str(self._activation.__name__) \
            + '_' + 'units' + '_' + str(self._num_units)
        if self._input_fourier:
            cell_str += '_input_fourier_'
        elif self._input_hilbert:
            cell_str += '_input_hilbert_'
        elif self._input_split_matmul:
            cell_str += '__input_split_matmul_'
        cell_str += '_stfl_' + str(self._stiefel)
        if self._real is False and self._single_gate is False:
            cell_str += '_ga_' + self._gate_activation.__name__
        if self._single_gate:
            cell_str += '_sg_'
        if self._real:
            cell_str += '_real_'
            if self._real_double:
                cell_str += '_realDouble_'
                cell_str += '_ga_' + self._gate_activation.__name__
        return cell_str

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        if self._output_size is None:
            if self._real:
                return self._num_units
            else:
                return self._num_units*2
        else:
            return self._output_size

    def zero_state(self, batch_size, dtype=tf.float32):
        if self._real:
            out = tf.zeros([batch_size, self._output_size], dtype=tf.float32)
            first_state = tf.zeros([batch_size, self._num_units])
        else:
            first_state = tf.complex(tf.zeros([batch_size, self._num_units]),
                                     tf.zeros([batch_size, self._num_units]))
            if self._output_size:
                if not self._complex_output:
                    out = tf.zeros([batch_size, self._output_size], dtype=tf.float32)
                else:
                    out = tf.zeros([batch_size, self._output_size], dtype=tf.complex64)
            else:
                if not self._complex_output:
                    out = tf.zeros([batch_size, self._num_units*2], dtype=tf.float32)
                else:
                    out = tf.zeros([batch_size, self._num_units], dtype=tf.complex64)
        return LSTMStateTuple(out, first_state)

    def double_memory_gate(self, h, x, scope, bias_init=4.0):
        """
        Complex GRU gates, the idea is that gates should make use of phase information.
        """

        with tf.variable_scope(scope, self._reuse):
            if self._real:
                ghr = matmul_plus_bias(h, self._num_units, scope='ghr', reuse=self._reuse,
                                       bias=False)
                gxr = matmul_plus_bias(x, self._num_units, scope='gxr', reuse=self._reuse,
                                       bias=True, bias_init=bias_init)
                gr = ghr + gxr
                r = tf.nn.sigmoid(gr)
                ghz = matmul_plus_bias(h, self._num_units, scope='ghz', reuse=self._reuse,
                                       bias=False)
                gxz = matmul_plus_bias(x, self._num_units, scope='gxz', reuse=self._reuse,
                                       bias=True, bias_init=bias_init)
                gz = ghz + gxz
                z = tf.nn.sigmoid(gz)

                if self._real_double:
                    ghr2 = matmul_plus_bias(h, self._num_units, scope='ghr2',
                                            reuse=self._reuse, bias=False)
                    gxr2 = matmul_plus_bias(x, self._num_units, scope='gxr2',
                                            reuse=self._reuse, bias=True,
                                            bias_init=bias_init)
                    gr2 = ghr2 + gxr2
                    r = self._gate_activation([gr, gr2], 'r', self._reuse)
                    ghz2 = matmul_plus_bias(h, self._num_units, scope='ghz2',
                                            reuse=self._reuse, bias=False)
                    gxz2 = matmul_plus_bias(x, self._num_units, scope='gxz2',
                                            reuse=self._reuse, bias=True,
                                            bias_init=bias_init)
                    gz2 = ghz2 + gxz2
                    z = self._gate_activation([gz, gz2], 'z', self._reuse)

            else:
                ghr = complex_matmul(h, self._num_units, scope='ghr', reuse=self._reuse)
                gxr = complex_matmul(x, self._num_units, scope='gxr', reuse=self._reuse,
                                     bias=True, bias_init_c=bias_init,
                                     bias_init_r=bias_init)
                gr = ghr + gxr
                r = self._gate_activation(gr, 'r', self._reuse)
                ghz = complex_matmul(h, self._num_units, scope='ghz', reuse=self._reuse)
                gxz = complex_matmul(x, self._num_units, scope='gxz', reuse=self._reuse,
                                     bias=True, bias_init_c=bias_init,
                                     bias_init_r=bias_init)
                gz = ghz + gxz
                z = self._gate_activation(gz, 'z', self._reuse)
            return r, z

    def single_memory_gate(self, h, x, scope, bias_init):
        """
        Use the real and imaginary parts of the gate equation to do the gating.
        """
        with tf.variable_scope(scope, self._reuse):
            if self._real:
                raise ValueError('Real cells cannot be single gated.')
            else:
                ghs = complex_matmul(h, self._num_units, scope='ghs', reuse=self._reuse)
                gxs = complex_matmul(x, self._num_units, scope='gxs', reuse=self._reuse,
                                     bias=True, bias_init_c=bias_init,
                                     bias_init_r=bias_init)
                gs = ghs + gxs
                return (tf.complex(tf.nn.sigmoid(tf.real(gs)),
                                   tf.zeros_like(tf.real(gs))),
                        tf.complex(tf.nn.sigmoid(tf.imag(gs)),
                                   tf.zeros_like(tf.imag(gs))))

    def __call__(self, inputs, state, scope=None):
        """
        Evaluate the cell equations.
        Params:
            inputs: The input values.
            state: the past cell state.
        Returns:
            output and new cell state touple.
        """
        # print('input_dtype', inputs.dtype)
        with tf.variable_scope("ComplexGatedRecurrentUnit", reuse=self._reuse):
            _, last_h = state

            if not self._real:
                if not self._complex_input:
                    if self._input_fourier:
                        cinputs = tf.complex(inputs, tf.zeros_like(inputs))
                        inputs = tf.fft(cinputs)
                    elif self._input_hilbert:
                        cinputs = tf.complex(inputs, tf.zeros_like(inputs))
                        inputs = hilbert(cinputs)
                    elif self._input_split_matmul:
                        # Map the inputs from R to C.
                        cinr = matmul_plus_bias(inputs, self._num_units,
                                                'real', self._reuse)
                        cini = matmul_plus_bias(inputs, self._num_units,
                                                'imag', self._reuse)
                        inputs = tf.complex(cinr, cini)
                    else:
                        inputs = tf.complex(inputs, tf.zeros_like(inputs))

            if self._dropout:
                print('adding dropout!')
                inputs = complex_dropout(inputs, 0.2)

            # use open gates initially when working with stiefel optimization.
            if self._stiefel:
                bias_init = 4.0
            else:
                bias_init = 0.0

            if self._single_gate:
                r, z = self.single_memory_gate(last_h, inputs, 'single_memory_gate',
                                               bias_init=bias_init)
            else:
                r, z = self.double_memory_gate(last_h, inputs, 'double_memory_gate',
                                               bias_init=bias_init)

            with tf.variable_scope("canditate_h"):
                if self._real:
                    cinWx = matmul_plus_bias(inputs, self._num_units, 'wx', bias=False,
                                             reuse=self._reuse)
                    rhU = matmul_plus_bias(tf.multiply(r, last_h), self._num_units, 'rhu',
                                           bias=True, orthogonal=self._stiefel,
                                           reuse=self._reuse)
                    tmp = cinWx + rhU
                else:
                    cinWx = complex_matmul(inputs, self._num_units, 'wx', bias=False,
                                           reuse=self._reuse)
                    rhU = complex_matmul(tf.multiply(r, last_h), self._num_units, 'rhu',
                                         bias=True, unitary=self._stiefel,
                                         reuse=self._reuse)
                    tmp = cinWx + rhU

                h_bar = self._activation(tmp)

                if self._dropout:
                    print('adding dropout!')
                    h_bar = complex_dropout(h_bar, 0.25)

            new_h = (1 - z)*last_h + z*h_bar

            if self._output_size:
                print('using an output projection.')
                if self._real:
                    output = matmul_plus_bias(new_h, self._output_size, 'out_map',
                                              reuse=self._reuse)
                else:
                    if self._complex_output:
                        output = complex_matmul(new_h, self._output_size, bias=True,
                                                scope='m_out', reuse=self._reuse)
                    else:
                        print('C to R mapping.')
                        output = C_to_R(new_h, self._output_size, reuse=self._reuse)
            else:
                if not self._complex_output:
                    print('real concatinated cell output')
                    output = tf.concat([tf.real(new_h), tf.imag(new_h)], axis=-1)
                else:
                    output = new_h
            newstate = LSTMStateTuple(output, new_h)
            return output, newstate
