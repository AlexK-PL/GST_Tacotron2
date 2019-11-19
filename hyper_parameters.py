from text import symbols

# creating a python dictionary with all hyper parameters

tacotron_params = {'filter_length': 1024,  # audio parameters:
                   'hop_length': 256,
                   'win_length': 1024,
                   'n_mel_channels': 80,
                   'mel_fmin': 0.0,
                   'mel_fmax': 8000.0,
                   'sampling_rate': 22050,
                   'max_wav_value': 32768.0,
                   'clipping_value': 1e-5,
                   'C': 1,
                   # dataset parameters:
                   'load_mel_from_disk': False,
                   'sort_by_length': False,
                   'text_cleaners': ['english_cleaners'],
                   # embedding parameters:
                   'symbols_embedding_length': 512,
                   'n_symbols': len(symbols),
                   # encoder parameters:
                   'encoder_embedding_dim': 512,
                   'encoder_convs': 3,
                   'conv_kernel_size': 5,
                   'conv_stride': 1,
                   'conv_dilation': 1,
                   'w_init_gain': 'relu',
                   # decoder parameters:
                   'number_frames_step': 1,
                   'decoder_rnn_dim': 1024,
                   'prenet_dim': 256,
                   'max_decoder_steps': 1000,
                   'gate_threshold': 0.5,  # Need to be reviewed
                   'p_attention_dropout': 0.1,
                   'p_decoder_dropout': 0.1,
                   # attention parameters:
                   'attention_rnn_dim': 1024,
                   'attention_dim': 128,
                   # location features parameters:
                   'attention_location_n_filters': 32,
                   'attention_location_kernel_size': 31,
                   # postnet parameters:
                   'postnet_embedding_dim': 512,
                   'postnet_kernel_size': 5,
                   'postnet_n_convolutions': 5,
                   # GST parameters:
                   'E': 512,
                   'token_num': 3,
                   'num_heads': 1,
                   'seq_ref_enc_filter_size': [3, 7, 11],  # phoneme, word/silence, utterance levels respectively
                   'ref_enc_out_channels': [8, 16, 16],
                   # optimization parameters:
                   'use_saved_learning_rate': True,
                   'batch_size': 32,  # 64 should be larger than the number of GPUs. Integer multiple of the num. of GPUs
                   'learning_rate': 1e-3,
                   'weight_decay': 1e-6,
                   'grad_clip_thresh': 1.0,
                   'mask_padding': False,
                   # experiment parameters:
                   'epochs': 300,  # 160, 500
                   'iters_per_checkpoint': 1500,  # 1000. How many iterations before validating
                   'seed': 1234,
                   'dynamic_loss_scaling': True,  # CHECK IT OUT!
                   'distributed_run': False,
                   'dist_backend': 'nccl',
                   'dist_url': "/home/alex/PyTorch_TACOTRON_2/pycharm-tacotron2",  # CHECK IT OUT!
                   'cudnn_enabled': True,
                   'cudnn_benchmark': False,
                   'fp16_run': False}
