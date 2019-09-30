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
  def load(self, pb_model_path, model_name='Tacotron'):
    log('Constructing model: %s' % model_name)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 28
    config.inter_op_parallelism_threads = 1

    self.graph = tf.Graph()
    self.session = tf.InteractiveSession(graph = self.graph, config=config)

    with self.graph.as_default():
      graph_def = tf.GraphDef()
      with tf.gfile.GFile(pb_model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    # print op in the graph
    ops = self.graph.get_operations()
    for op in ops:
        print('- {0:20s} "{1}" ({2} outputs)'.format(op.type, op.name, len(op.outputs)))

    self.inputs = self.graph.get_tensor_by_name('import/Input/input_x:0')
    self.input_lengths = self.graph.get_tensor_by_name('import/Input/input_seq_len:0')
    self.speaker_id = self.graph.get_tensor_by_name('import/Input/speaker_id:0')
    self.output_tensor = self.graph.get_tensor_by_name("import/Output/predict/BiasAdd:0")


  def synthesize(self, texts):
    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # create profiler
    profiler = tf.profiler.Profiler(self.session.graph)
    step = -1

    feed_dict = {
      self.inputs: texts,
      self.input_lengths: np.asarray([texts.shape[1]], dtype=np.int32),
      self.speaker_id: np.asarray([0], dtype=np.int32),
    }

    run_metadata = tf.RunMetadata()
    log("start session run=====================================")
    start = time.time()
    pred = self.session.run(self.output_tensor, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
    end = time.time() - start
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
    log("Profiler operations done!=====================================")
    return pred


def tacotron_synthesize(pb_model_path, texts):
  synth = Synthesizer()
  synth.load(pb_model_path)
  # warmup
  log("warmup run=====================================")
  _ = synth.synthesize(texts)

  log("start running sentence=====================================")
  log("input shape:{}".format(texts.shape))
  log("input value:{}".format(texts))

  pred = synth.synthesize(texts)

  log("output shape:{}".format(pred.shape))
  log("output value:{}".format(pred))
  log('end running sentence=====================================')


def main():

  taco_pb_model = 'aco.pb'
  mode = 'eval'
  model = 'Tacotron'

  # Embeddings ==> [batch_size, sequence_length, embedding_dim]
  sequence_length = 32
  embedding_dim = 563
  embeddings = np.random.random([1, sequence_length, embedding_dim])

  tacotron_synthesize(taco_pb_model, embeddings)

if __name__ == '__main__':
  main()