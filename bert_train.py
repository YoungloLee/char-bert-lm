from bert import BERTModel, dataset, WarmUpSchedule, create_masks
from loss import loss_function
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


# loading
with open('./syllable_tokenizer_all.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Mappings from symbol to numeric ID and vice versa:

tokens = [x[0] for x in sorted(tokenizer.word_counts.items(), key=lambda item: item[1], reverse=True)]

PAD = '<p>'
SOS = '<s>'
EOS = '</s>'

tokens = [PAD] + tokens
# start and end of utterance (for LAS)
tokens.append(SOS)
tokens.append(EOS)

index_token = {idx: ch for idx, ch in enumerate(tokens)}
token_index = {ch: idx for idx, ch in enumerate(tokens)}

num_classes = len(token_index)
num_layers = 4
d_model = 256
dff = 1024
num_heads = 8
dropout_rate = 0.1
batch_size = 64

lang_model = BERTModel(num_classes, num_layers, d_model, dff, num_heads, dropout_rate)

train_data, train_steps = dataset('./preprocess.txt', batch_size)
valid_data, valid_steps = dataset('./preprocess_light.txt', batch_size, mode='valid')

save_dir = os.path.join('./pretrained')

learning_rate = WarmUpSchedule(d_model, int(train_steps * 5))
opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

temp_learning_rate = WarmUpSchedule(d_model, int(train_steps * 5))
plt.plot(temp_learning_rate(tf.range(int(train_steps * 5), dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()

checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, transformer=lang_model.model)
manager = tf.train.CheckpointManager(checkpoint, directory=save_dir, max_to_keep=5)

checkpoint_state = tf.train.get_checkpoint_state(save_dir)
if checkpoint_state and checkpoint_state.model_checkpoint_path:
    print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
    checkpoint.restore(manager.latest_checkpoint)
else:
    print('No model to load at {}'.format(save_dir))
    print('Starting new training!')
eval_best_loss = np.inf

summary_list = list()
lang_model.model.summary(line_length=180, print_fn=lambda x: summary_list.append(x))
for summary in summary_list:
    print(summary)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_acc')

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int32)]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    combined_mask = create_masks(inp)
    with tf.GradientTape() as tape:
        predictions = lang_model.model(inp, combined_mask, True)
        loss = loss_function(tar, predictions)
    gradients = tape.gradient(loss, lang_model.model.trainable_variables)
    opt.apply_gradients(zip(gradients, lang_model.model.trainable_variables))

    tar_weight = tf.cast(tf.logical_not(tf.math.equal(tar, 0)), tf.int32)
    train_loss(loss)
    train_acc(tar, predictions, sample_weight=tar_weight)


@tf.function(input_signature=train_step_signature)
def valid_step(inp, tar):
    combined_mask = create_masks(inp)
    predictions = lang_model.model(inp, combined_mask, False)
    loss = loss_function(tar, predictions)

    tar_weight = tf.cast(tf.logical_not(tf.math.equal(tar, 0)), tf.int32)
    valid_loss(loss)
    valid_acc(tar, predictions, sample_weight=tar_weight)


# Train
for epoch in range(100):
    train_loss.reset_states()
    train_acc.reset_states()
    valid_loss.reset_states()
    valid_acc.reset_states()

    for (batch, (input, label)) in enumerate(train_data):
        train_step(input, label)
        message = '[Epoch {:.3f}] [Step {:7d}] [loss={:.5f}, acc={:.5f}]'.format(
            epoch + (batch / train_steps), int(checkpoint.step), train_loss.result(), train_acc.result())
        print(message)
        checkpoint.step.assign_add(1)

    for (batch, (input, label)) in enumerate(valid_data):
        valid_step(input, label)

    print('Eval loss & ler & acc for global step {}: {:.3f}, {:.3f}'.format(
        int(checkpoint.step), valid_loss.result(), valid_acc.result()))

    if valid_loss.result() < eval_best_loss:
        # Save model and current global step
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
        print('Validation loss improved from {:.2f} to {:.2f}'.format(eval_best_loss, valid_loss.result()))
        eval_best_loss = valid_loss.result()

