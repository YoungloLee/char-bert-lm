from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Layer, Embedding
from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf
from bert_train import token_index


def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_masks(inp):
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
    enc_padding_mask = create_padding_mask(inp)
    combined_mask = tf.maximum(enc_padding_mask, look_ahead_mask)
    return combined_mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = Dense(d_model)
        self.ffn_intermediate = Dense(dff, activation=gelu)
        self.ffn2 = Dense(d_model)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        attn_output, _ = self.mha(x, x, x, look_ahead_mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.ffn1(attn_output)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        intermediate_output = self.ffn_intermediate(out1)  # (batch_size, input_seq_len, d_model)

        out2 = self.ffn2(intermediate_output)
        out2 = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out2 + out1)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class LanguageModel(Layer):
    def __init__(self, d_model, vocab_size):
        super(LanguageModel, self).__init__()

        self.d_model = d_model

        self.dense = Dense(d_model, activation=gelu)
        self.layernorm = LayerNormalization(epsilon=1e-6)

        self.output_bias = self.add_weight("output_bias", shape=vocab_size, initializer=tf.zeros_initializer())

    def call(self, x, output_weights, training):
        x = self.dense(x)
        x = self.layernorm(x)

        logit = tf.matmul(x, output_weights, transpose_b=True)
        logit = tf.nn.bias_add(logit, self.output_bias)

        return logit

    def gather_indexes(self, x, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        width = tf.shape(x)[2]

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(x, [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor


class BERT(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_input, rate=0.1):
        super(BERT, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size, pe_input, rate)

        self.decoder = LanguageModel(d_model, vocab_size)

    def call(self, inp, look_ahead_mask, training):
        enc_output = self.encoder(inp, training, look_ahead_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        output_weights = self.encoder.embedding.weights[0]
        final_output = self.decoder(enc_output, output_weights, training)

        return final_output


class BERTModel:
    def __init__(self, num_classes, num_layers, d_model, dff, num_heads, dropout_rate):
        self.encoder, self.decoder, self.model = None, None, None
        self.generate_model(num_classes, num_layers, d_model, dff, num_heads, dropout_rate)

    def generate_model(self, num_classes, num_layers, d_model, dff, num_heads, dropout_rate):

        model = BERT(num_layers, d_model, num_heads, dff, vocab_size=num_classes, pe_input=10000, rate=dropout_rate)

        temp_input = tf.random.uniform((64, 40), dtype=tf.int32, minval=0, maxval=70)

        fn_out = model(temp_input, look_ahead_mask=None, training=False)

        print("Transformer encoder input shape: (batch_size, enc_length) {}".format(temp_input.shape))
        print("Transformer result shape: (batch_size, enc_length, target_vocab_size) {}".format(fn_out.shape))

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.model = model


def dataset(fname, batch_size, mode='train'):
    with open(fname, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines if x is not '\n']
    steps = int(np.ceil(len(lines) / batch_size))

    def encode(input):
        # Collect signals
        input = input.numpy().decode('utf8')
        sequence = np.array([token_index[SOS]] + [token_index[x] for x in input] + [token_index[EOS]]).astype('int32')
        return sequence

    def tf_encode(input):
        result_sequence = tf.py_function(encode, [input], tf.int32)
        result_sequence.set_shape([None])
        return result_sequence[:-1], result_sequence[1:]

    data = tf.data.Dataset.from_tensor_slices(lines)
    data = data.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if mode == 'train':
        data = data.shuffle(10000).padded_batch(batch_size, padded_shapes=([None], [None]))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        data = data.padded_batch(batch_size, padded_shapes=([None], [None]))

    return data, steps


class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=1000000):
        super(WarmUpSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)