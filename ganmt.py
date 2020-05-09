import tensorflow as tf
import numpy as np

class ganmt(object):
   def __init__(
      self,
      session,
      in_vocab_size=30000,
      out_vocab_size=30000,
      sentence_length=50,
      learning_rate=1e-4
   ):
      self.session = session
      self.learning_rate = learning_rate
      self.sentence_length = sentence_length
      self.in_vocab_size = in_vocab_size
      self.out_vocab_size = out_vocab_size

      self.input_sequence = tf.placeholder(
         shape=[None, self.sentence_length, self.in_vocab_size]
         dtype=tf.float32
      )

   def eval_generator(self, input_layer, output_size, scope_name="generator"):
      vocab_size = tf.shape(input_layer)[2]
      sentence_length = tf.shape(input_layer)[1]
      embedding_size = vocab_size//10

      with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
         W_embedding = tf.get_variable(
            name="embedding_weight",
            shape=[vocab_size, embedding_size],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal()
         )
         b_embedding = tf.get_variable(
            name="embedding_bias",
            shape=[embedding_size],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal()
         )

         embedding_layer_flat = tf.nn.relu(
            tf.matmul(
               tf.reshape(input_layer, [-1, vocab_size]),
               W_embedding
            ) + b_embedding
         )
         embedding_layer = tf.reshape(
            embedding_layer_flat,
            [-1, sentence_length, embedding_size, 1]
         )

         layer_num = 1
         conv_layer_1 = conv_layer(embedding_layer, layer_num, w=7, h=3, c_in=1, c_out=32)
         layer_num += 1

         conv_layer_2 = conv_layer(conv_layer_2, layer_num, w=7, h=3, c_in=32, c_out=32)
         layer_num += 1

         conv_layer_3 = conv_layer(conv_layer_3, layer_num, w=7, h=3, c_in=32, c_out=8)
         layer_num += 1

         W_layer_out = tf.get_variable(
            name="output_weight",
            shape=[8*embedding_size, output_size],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal()
         )
         b_layer_out = tf.get_variable(
            name="output_bias",
            shape=[output_size],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal()
         )

         logits_out_flat = tf.matmul(
            tf.reshape(
               conv_layer_3, [-1, 8*embedding_size]
            ),
            W_layer_out
         ) + b_layer_out
         predictions_flat = tf.nn.softmax(logits_out_flat)

         #logits_out = tf.reshape(logits_out_flat, [-1, sentence_length, output_size])
         predictions_out = tf.reshape(predictions_flat, [-1, sentence_length, output_size])

         #return logits_out, logits_out_flat
         return predictions_out

   def eval_discriminator(self, input_layer, scope_name="discriminator_eval"):
      sentence_length = tf.shape(input_layer)[1]
      vocab_size = tf.shape(input_layer)[2]
      with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
         input_4d = tf.reshape(input_layer, [-1, sentence_length, vocab_size, 1])

         W_embedding = tf.get_variable(
            name="embedding_weight",
            shape=[vocab_size, embedding_size],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal()
         )
         b_embedding = tf.get_variable(
            name="embedding_bias",
            shape=[embedding_size],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal()
         )

         embedding_layer_flat = tf.nn.relu(
            tf.matmul(
               tf.reshape(input_layer, [-1, vocab_size]),
               W_embedding
            ) + b_embedding
         )
         embedding_layer = tf.reshape(
            embedding_layer_flat,
            [-1, sentence_length, embedding_size, 1]
         )

         layer_num = 1
         conv_layer_1 = conv_layer(embedding_layer, layer_num, w=7, h=3, c_in=1, c_out=32)
         layer_num += 1

         max_pool_layer_1 = tf.nn.max_pool2d(conv_layer_1, ksize=2, strides=2)

         conv_layer_2 = conv_layer(max_pool_layer_1, layer_num, w=7, h=3, c_in=32, c_out=16)
         layer_num += 1

         max_pool_layer_2 = tf.nn.max_pool2d(conv_layer_2, ksize=2, strides=2)

         conv_layer_3 = conv_layer(max_pool_layer_2, layer_num, w=7, h=3, c_in=16, c_out=8)
         layer_num += 1

         max_pool_layer_3 = tf.nn.max_pool2d(conv_layer_3, ksize=2, strides=2)

         conv_layer_4 = conv_layer(max_pool_layer_3, layer_num, w=7, h=3, c_in=8, c_out=1)
         layer_num += 1

         W_output = tf.get_variable(
            name="W_output",
            shape=[75*12, 1]
         )

         b_output = tf.get_variable(
            name="b_output",
            shape=[1,]
         )

         fc_layer = tf.matmul(
            tf.reshape(conv_layer_4, [-1, 75*12]),
            W_output
         ) + b_output

         return fc_layer

   def conv_layer(self, feature_map, layer_num, c_in, c_out, w=3, h=3, scope_name="conv_layer"):
      fm_size = tf.shape(feature_map)[1:3]
      with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
         W_conv = tf.get_variable(
            name="conv_W_" + str(layer_name),
            shape=[w, h, c_in, c_out],
            initializer=tf.initializers.random_normal()
         )
         b_conv = tf.get_variable(
            name="conv_b_" + str(layer_name),
            shape=[*fm_size, c_out],
            initializer=tf.initializers.random_normal()
         )

         conv_out = tf.nn.conv2d(feature_map, W_conv, padding="same")
         layer_out = tf.nn.relu(
            conv_out + b_conv
         )

         return layer_out
