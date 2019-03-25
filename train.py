import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_train, directories

def train():

    start_time = time.time()
    G_loss_best, D_loss_best = float('inf'), float('inf')
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    print('Training on dataset', args.dataset)
    if config.use_conditional_GAN:
        print('Using conditional GAN')
        paths, semantic_map_paths = Data.load_dataframe(directories.train, load_semantic_maps=True)
        test_paths, test_semantic_map_paths = Data.load_dataframe(directories.test, load_semantic_maps=True)
    else:
        paths = Data.load_dataframe(directories.train)
        test_paths = Data.load_dataframe(directories.test)

    # Build graph
    gan = Model(config, paths, name=args.name, dataset=args.dataset)
    saver = tf.train.Saver()

    if config.use_conditional_GAN:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths,
                               gan.test_semantic_map_path_placeholder: test_semantic_map_paths}
        feed_dict_train_init = {gan.path_placeholder: paths,
                                gan.semantic_map_path_placeholder: semantic_map_paths}
    else:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths}
        feed_dict_train_init = {gan.path_placeholder: paths}

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(gan.train_iterator.string_handle())
        test_handle = sess.run(gan.test_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('{} restored.'.format(args.restore_path))

        sess.run(gan.test_iterator.initializer, feed_dict=feed_dict_test_init)

        for epoch in range(config.num_epochs):

            sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_train_init)

            # Run diagnostics
            G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, directories, sess, saver, train_handle,
                start_time, epoch, args.name, G_loss_best, D_loss_best)

            while True:
                try:
                    # Update generator
                    # for _ in range(8):
                    feed_dict = {gan.training_phase: True, gan.handle: train_handle}
                    sess.run(gan.G_train_op, feed_dict=feed_dict)

                    # Update discriminator
                    step, _ = sess.run([gan.D_global_step, gan.D_train_op], feed_dict=feed_dict)

                    if step % config.diagnostic_steps == 0:
                        G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, directories, sess, saver, train_handle,
                            start_time, epoch, args.name, G_loss_best, D_loss_best)
                        Utils.single_plot(epoch, step, sess, gan, train_handle, args.name, config)
                        # for _ in range(4):
                        #    sess.run(gan.G_opt_op, feed_dict=feed_dict)


                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        '{}_last.ckpt'.format(args.name)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               '{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

train() 
