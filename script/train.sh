set -e
cd ..

MODEL=AITL_20_4
MODEL_DIR=model/$MODEL
EVAL_MODEL='1_2_3_4_5_6_7_8_9_10_11_12_13'

CUDA_VISIBLE_DEVICES=$1 python dynamic_train.py \
    --output_dir=$MODEL_DIR \
    --AITL_predict_lambda=-15 \
    --AITL_predict_num_steps=1 \
    --AITL_predict_num_seeds=1 \
    --AITL_encoder_vocab_size=20 \
    --AITL_decoder_vocab_size=20 \
    --AITL_encoder_length=4 \
    --AITL_decoder_length=4 \
    --AITL_encoder_emb_size=128 \
    --AITL_encoder_num_layers=3 \
    --AITL_encoder_hidden_size=128 \
    --AITL_encoder_dropout=0.1 \
    --AITL_mlp_num_layers=5 \
    --AITL_mlp_hidden_size=128 \
    --AITL_mlp_dropout=0.1 \
    --AITL_decoder_num_layers=1 \
    --AITL_decoder_hidden_size=128 \
    --AITL_decoder_dropout=0.0 \
    --AITL_image_hidden_size=128 \
    --AITL_train_epochs=1 \
    --AITL_optimizer=adam \
    --AITL_lr=0.00005 \
    --AITL_batch_size=64 \
    --AITL_save_frequency=100 \
    --AITL_trade_off=0.8 \
    --AITL_num_sample_eval=50 \
    --AITL_num_sample_append=100 \
    --AITL_model_id=1 \
    --attack_max_epsilon=16 \
    --attack_image_height=299 \
    --attack_image_width=299 \
    --attack_image_resize=330 \
    --attack_model=1 \
    --attack_num_iter=10 \
    --attack_num_noise=1 \
    --attack_untarget=True \
    --attack_init_num_sample=100 \
    --attack_pool_num_sample=100 \
    --evaluate_model=${EVAL_MODEL}
