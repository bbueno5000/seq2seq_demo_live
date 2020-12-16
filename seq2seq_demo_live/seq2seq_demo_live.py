"""
We give encoder input sequence like "hello how are you",
we take the last hidden state and feed to decoder
and it will generate a decoded value.
We compare that to target value,
if translation would be "bonjour ca va"
and minimize the difference by optimizing a loss function.

We will teach our model to memorize and reproduce input sequence. 
Sequences will be random, with varying length.
Since random sequences do not contain any structure, 
model will not be able to exploit any patterns in data. 
It will simply encode sequence in a thought vector, then decode from it.
This is not about prediction, it's about understanding this architecture.

The encoder is bidrectional so it feeds previously generated tokens
during training as inputs, instead of target sequence.
"""
import matplotlib.pyplot as pyplot
import numpy
import tensorflow

class Helpers:
    """
    DOCSTRING
    """
    def batch(self, inputs, max_sequence_length=None):
        """
        batch function
        Args:
            inputs:
                list of sentences (integer lists)
            max_sequence_length:
                integer specifying how large should `max_time` dimension be.
                If None, maximum sequence length would be used.
        Returns:
            inputs_time_major:
                input sentences transformed into time-major matrix 
                (shape [max_time, batch_size]) padded with 0s
            sequence_lengths:
                batch-sized list of integers specifying amount of active 
                time steps in each input sequence
        """
        sequence_lengths = [len(seq) for seq in inputs]
        batch_size = len(inputs)
        if max_sequence_length is None:
            max_sequence_length = max(sequence_lengths)
        inputs_batch_major = numpy.zeros(shape=[batch_size, max_sequence_length], dtype=numpy.int32)
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element
        inputs_time_major = inputs_batch_major.swapaxes(0, 1)
        return inputs_time_major, sequence_lengths

    def random_sequences(self, length_from, length_to, vocab_lower, vocab_upper, batch_size):
        """
        random_sequences
        Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
        """
        if length_from > length_to:
                raise ValueError('length_from > length_to')
        def random_length():
            if length_from == length_to:
                return length_from
            return numpy.random.randint(length_from, length_to + 1)
        while True:
            yield [
                numpy.random.randint(
                low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
                for _ in range(batch_size)]

class Seq2Seq:
    """
    DOCSTRING
    """
    def __call__(self):
        """
        DOCSTRING
        """
        tensorflow.reset_default_graph()
        sess = tensorflow.InteractiveSession()
        PAD = 0
        EOS = 1
        vocab_size = 10
        input_embedding_size = 20
        encoder_hidden_units = 20
        decoder_hidden_units = encoder_hidden_units
        encoder_inputs = tensorflow.placeholder(shape=(None, None), dtype=tensorflow.int32, name='encoder_inputs')
        encoder_inputs_length = tensorflow.placeholder(shape=(None,), dtype=tensorflow.int32, name='encoder_inputs_length')
        decoder_targets = tensorflow.placeholder(shape=(None, None), dtype=tensorflow.int32, name='decoder_targets')
        embeddings = tensorflow.Variable(tensorflow.random_uniform(
            [vocab_size, input_embedding_size], -1.0, 1.0), dtype=tensorflow.float32)
        encoder_inputs_embedded = tensorflow.nn.embedding_lookup(embeddings, encoder_inputs)
        encoder_cell = tensorflow.python.ops.rnn_cell.LSTMCell(encoder_hidden_units)
        ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = (
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell, cell_bw=encoder_cell, inputs=encoder_inputs_embedded,
                sequence_length=encoder_inputs_length, dtype=tensorflow.float64, time_major=True))
        encoder_outputs = tensorflow.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        encoder_final_state_c = tensorflow.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
        encoder_final_state_h = tensorflow.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
        encoder_final_state = tensorflow.python.ops.rnn_cell.LSTMStateTuple(
            c=encoder_final_state_c, h=encoder_final_state_h)
        decoder_cell = tensorflow.python.ops.rnn_cell.LSTMCell(decoder_hidden_units)
        encoder_max_time, batch_size = tensorflow.unstack(tensorflow.shape(encoder_inputs))
        decoder_lengths = encoder_inputs_length + 3
        W = tensorflow.Variable(tensorflow.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tensorflow.float32)
        b = tensorflow.Variable(tensorflow.zeros([vocab_size]), dtype=tensorflow.float32)
        assert EOS == 1 and PAD == 0
        eos_time_slice = tensorflow.ones([batch_size], dtype=tensorflow.int32, name='EOS')
        pad_time_slice = tensorflow.zeros([batch_size], dtype=tensorflow.int32, name='PAD')
        eos_step_embedded = tensorflow.nn.embedding_lookup(embeddings, eos_time_slice)
        pad_step_embedded = tensorflow.nn.embedding_lookup(embeddings, pad_time_slice)
        decoder_outputs_ta, decoder_final_state, _ = tensorflow.nn.raw_rnn(decoder_cell, self.loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()
        decoder_max_steps, decoder_batch_size, decoder_dim = tensorflow.unstack(tensorflow.shape(decoder_outputs))
        decoder_outputs_flat = tensorflow.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tensorflow.add(tensorflow.matmul(decoder_outputs_flat, W), b)
        decoder_logits = tensorflow.reshape(
            decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
        decoder_prediction = tensorflow.argmax(decoder_logits, 2)
        stepwise_cross_entropy = tensorflow.nn.softmax_cross_entropy_with_logits(
            labels=tensorflow.one_hot(decoder_targets, depth=vocab_size, dtype=tensorflow.float32), logits=decoder_logits)
        loss = tensorflow.reduce_mean(stepwise_cross_entropy)
        train_op = tensorflow.train.AdamOptimizer().minimize(loss)
        sess.run(tensorflow.global_variables_initializer())
        batch_size = 100
        batches = Helpers.random_sequences(
            length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)
        print('head of the batch:')
        for seq in next(batches)[:10]:
            print(seq)
        loss_track = list()
        max_batches = 3001
        batches_in_epoch = 1000
        try:
            for batch in range(max_batches):
                fd = self.next_feed()
                _, l = sess.run([train_op, loss], fd)
                loss_track.append(l)
                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                    predict_ = sess.run(decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        print('    input     > {}'.format(inp))
                        print('    predicted > {}'.format(pred))
                        if i >= 2:
                            break
                    print()
        except KeyboardInterrupt:
            print('training interrupted')
        pyplot.plot(loss_track)
        print('loss {:.4f} after {} examples (batch_size={})'.format(
            loss_track[-1], len(loss_track)*batch_size, batch_size))

    def loop_fn(self, time, previous_output, previous_state, previous_loop_state):
        """
        DOCSTRING
        """
        if previous_state is None:
            assert previous_output is None and previous_state is None
            return self.loop_fn_initial()
        else:
            return self.loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

    def loop_fn_initial(self):
        """
        DOCSTRING
        """
        initial_elements_finished = (0 >= decoder_lengths)
        initial_input = eos_step_embedded
        initial_cell_state = encoder_final_state
        initial_cell_output = None
        initial_loop_state = None
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

    def loop_fn_transition(self, time, previous_output, previous_state, previous_loop_state):
        """
        Attention mechanism
        Choose which previously generated token to pass as input in the next timestep.
        """
        def get_next_input():
            output_logits = tensorflow.add(tensorflow.matmul(previous_output, W), b)
            prediction = tensorflow.argmax(output_logits, axis=1)
            next_input = tensorflow.nn.embedding_lookup(embeddings, prediction)
            return next_input
        elements_finished = (time >= decoder_lengths)
        finished = tensorflow.reduce_all(elements_finished)
        input = tensorflow.cond(finished, lambda: pad_step_embedded, get_next_input)
        state = previous_state
        output = previous_output
        loop_state = None
        return (elements_finished, input, state, output, loop_state)

    def next_feed(self):
        """
        DOCSTRING
        """
        batch = next(batches)
        encoder_inputs_, encoder_input_lengths_ = Helpers.batch(batch)
        decoder_targets_, _ = Helpers.batch(
            [(sequence) + [EOS] + [PAD] * 2 for sequence in batch])
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_}

if __name__ == '__main__':
    seq2seq = Seq2Seq()
    seq2seq()
