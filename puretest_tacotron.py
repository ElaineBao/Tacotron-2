import argparse
import os
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from infolog import log
import numpy as np

from tensorflow.contrib.seq2seq import dynamic_decode
from tacotron.models.helpers import TacoTestHelper
from tacotron.models.modules import *
from tacotron.models.Architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from tacotron.models.custom_decoder import CustomDecoder
from tacotron.models.attention import LocationSensitiveAttention
from tacotron.utils.symbols import symbols
from tacotron.utils.text import text_to_sequence

class Synthesizer:
  def load(self, checkpoint_path, hparams, model_name='Tacotron'):
    log('Constructing model: %s' % model_name)
    #Force the batch size to be known in order to use attention masking in batch synthesis
    inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
    input_lengths = tf.placeholder(tf.int32, (None), name='input_lengths')
    split_infos = tf.placeholder(tf.int32, (hparams.tacotron_num_gpus, None), name='split_infos')
    with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
      self.model = Tacotron(hparams)
      self.model.initialize(inputs, input_lengths, split_infos=split_infos)

      self.mel_outputs = self.model.tower_mel_outputs
      self.linear_outputs = self.model.tower_linear_outputs
      self.alignments = self.model.tower_alignments
      self.stop_token_prediction = self.model.tower_stop_token_prediction

    self._hparams = hparams
    self.inputs = inputs
    self.input_lengths = input_lengths
    self.split_infos = split_infos

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = hparams.intra_op_parallelism_threads
    config.inter_op_parallelism_threads = hparams.inter_op_parallelism_threads

    self.session = tf.Session(config=config)
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)

  def synthesize(self, texts):
    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # create profiler
    profiler = tf.profiler.Profiler(self.session.graph)
    step = -1

    hparams = self._hparams
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

    input_seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
    input_lengths = [len(seq) for seq in input_seqs]
    split_infos = []
    max_seq_len = max([len(x) for x in input_seqs])
    split_infos.append([max_seq_len, 0, 0, 0])

    feed_dict = {
      self.inputs: input_seqs,
      self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
      self.split_infos: np.asarray(split_infos, dtype=np.int32),
    }

    run_metadata = tf.RunMetadata()
    log("start session run=====================================")
    start = time.time()
    linears, mels, alignments, stop_tokens = self.session.run([self.linear_outputs, self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    end = time.time() - start
    log("output mels:{}".format(mels))
    step += 1
    profiler.add_step(step, run_metadata)
    log("end session run, time:{}=====================================".format(end))

    option_builder = tf.profiler.ProfileOptionBuilder
    opts = (option_builder(option_builder.time_and_memory()).
            with_step(-1). # with -1, should compute the average of all registered steps.
            with_file_output('./profiling/inference_profiling.txt').
            select(['micros','bytes','occurrence']).order_by('micros').
            build())
    # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
    profiler.profile_operations(options=opts)
    log("Profiler operations done!================================")


class Tacotron():
  """Tacotron-2 Feature prediction Model.
  """
  def __init__(self, hparams):
    self._hparams = hparams

  def split_func(self, x, split_pos):
    rst = []
    start = 0
    # x will be a numpy array with the contents of the placeholder below
    for i in range(split_pos.shape[0]):
      rst.append(x[:,start:start+split_pos[i]])
      start += split_pos[i]
    return rst

  def initialize(self, inputs, input_lengths, gta=False,
      is_training=False, is_evaluating=False, split_infos=None):
    """
    Initializes the model for inference
    sets "mel_outputs" and "alignments" fields.
    Args:
      - inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      - input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
      of each sequence in inputs.
    """

    cpu_device = '/cpu:0'
    with tf.device(cpu_device):
      hp = self._hparams
      lout_int = [tf.int32]*hp.tacotron_num_gpus
      tower_input_lengths = tf.split(input_lengths, num_or_size_splits=hp.tacotron_num_gpus, axis=0)
      batch_size = tf.shape(inputs)[0]
      mel_channels = hp.num_mels
      linear_channels = hp.num_freq

      p_inputs = tf.py_func(self.split_func, [inputs, split_infos[:, 0]], lout_int)
      tower_inputs = []
      for i in range (hp.tacotron_num_gpus):
        tower_inputs.append(tf.reshape(p_inputs[i], [batch_size, -1]))

    self.tower_decoder_output = []
    self.tower_alignments = []
    self.tower_stop_token_prediction = []
    self.tower_mel_outputs = []
    self.tower_linear_outputs = []

    with tf.device(cpu_device):
      with tf.variable_scope('inference') as scope:
        # Embeddings ==> [batch_size, sequence_length, embedding_dim]
        self.embedding_table = tf.get_variable(
          'inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32)
        embedded_inputs = tf.nn.embedding_lookup(self.embedding_table, tower_inputs[i])

        #Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
        encoder_cell = TacotronEncoderCell(
          EncoderConvolutions(is_training, hparams=hp, scope='encoder_convolutions'),
          EncoderRNN(is_training, size=hp.encoder_lstm_units,
            zoneout=hp.tacotron_zoneout_rate, scope='encoder_LSTM'))

        encoder_outputs = encoder_cell(embedded_inputs, tower_input_lengths[i])

        #For shape visualization purpose
        enc_conv_output_shape = encoder_cell.conv_output_shape

        #Decoder Parts
        #Attention Decoder Prenet
        prenet = Prenet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate, scope='decoder_prenet')
        #Attention Mechanism
        attention_mechanism = LocationSensitiveAttention(hp.attention_dim, encoder_outputs, hparams=hp, is_training=is_training,
          mask_encoder=hp.mask_encoder, memory_sequence_length=tf.reshape(tower_input_lengths[i], [-1]), smoothing=hp.smoothing,
          cumulate_weights=hp.cumulative_weights)
        #Decoder LSTM Cells
        decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers,
          size=hp.decoder_lstm_units, zoneout=hp.tacotron_zoneout_rate, scope='decoder_LSTM')
        #Frames Projection layer
        frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step, scope='linear_transform_projection')
        #<stop_token> projection layer
        stop_projection = StopProjection(is_training or is_evaluating, shape=hp.outputs_per_step, scope='stop_token_projection')
        #Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
        decoder_cell = TacotronDecoderCell(prenet, attention_mechanism, decoder_lstm, frame_projection, stop_projection)
        #initial decoder state
        decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        #Define the helper for our decoder
        self.helper = TacoTestHelper(batch_size, hp)
        #Only use max iterations at synthesis time
        max_iters = hp.max_iters
        #Decode
        (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
          CustomDecoder(decoder_cell, self.helper, decoder_init_state),
          impute_finished=False, maximum_iterations=max_iters, swap_memory=False)
        # Reshape outputs to be one output per entry
        #==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
        decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hp.num_mels])
        stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

        #Postnet
        postnet = Postnet(is_training, hparams=hp, scope='postnet_convolutions')
        #Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
        residual = postnet(decoder_output)
        #Project residual to same dimension as mel spectrogram
        #==> [batch_size, decoder_steps * r, num_mels]
        residual_projection = FrameProjection(hp.num_mels, scope='postnet_projection')
        projected_residual = residual_projection(residual)

        #Compute the mel spectrogram
        mel_outputs = decoder_output + projected_residual

        #Grab alignments from the final decoder state
        alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

        self.tower_decoder_output.append(decoder_output)
        self.tower_alignments.append(alignments)
        self.tower_stop_token_prediction.append(stop_token_prediction)
        self.tower_mel_outputs.append(mel_outputs)

    log('Initialized Tacotron model. Dimensions (? = dynamic shape): ')
    log('  Train mode:               {}'.format(is_training))
    log('  Eval mode:                {}'.format(is_evaluating))
    log('  GTA mode:                 {}'.format(gta))
    log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
    log('  Input:                    {}'.format(inputs.shape))


def tacotron_synthesize(hparams, checkpoint, sentences):
  checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
  log('loaded model at {}'.format(checkpoint_path))

  synth = Synthesizer()
  synth.load(checkpoint_path, hparams)
  # warmup
  log("warmup run")
  synth.synthesize(sentences[0])
  for i, texts in enumerate(sentences):
      log("start runing sentence:{}".format(texts))
      start = time.time()
      synth.synthesize(texts)
      end = time.time() - start
      log('eval time for synthesize one sentence:{}'.format(end))


def main():

  taco_checkpoint = 'logs-Tacotron/taco_pretrained/'
  sentences = ['Scientists at the CERN laboratory say they have discovered a new particle.']
  mode = 'eval'
  model = 'Tacotron'

  hparams = tf.contrib.training.HParams(
    tacotron_synthesis_batch_size=1,
    tacotron_num_gpus=1,
    intra_op_parallelism_threads = 28,
    inter_op_parallelism_threads= 1,
    embedding_dim=512,
    tacotron_teacher_forcing_mode = 'scheduled', #Can be ('constant' or 'scheduled')
    cleaners='english_cleaners',

    #EncoderConvolutions params
    encoder_lstm_units = 256,
    enc_conv_num_layers = 3,
    enc_conv_kernel_size = (5, ),
    enc_conv_channels = 512,
    tacotron_dropout_rate=0.5,
    batch_norm_position = 'after',
    #EncoderRNN params
    tacotron_zoneout_rate=0.1,
    #Decoder
    prenet_layers=[256, 256],
    decoder_layers = 2,
    decoder_lstm_units = 1024,
    #Attention mechanism
    smoothing=False,
    mask_encoder = True,
    attention_dim=128,
    attention_filters=32,
    attention_kernel = (31, ),
    cumulative_weights=True,
    synthesis_constraint = False,
	  synthesis_constraint_type = 'window',
	  attention_win_size = 7,
    #Frames Projection layer
    num_mels=80,
    num_freq = 1025,
    outputs_per_step=1,
    stop_at_any = True,
    #Residual postnet
    postnet_num_layers = 5,
    postnet_kernel_size = (5, ),
    postnet_channels = 512,
    max_iters = 1,
  )

  tacotron_synthesize(hparams, taco_checkpoint, sentences)

if __name__ == '__main__':
  main()