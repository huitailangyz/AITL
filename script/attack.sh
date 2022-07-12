set -e
cd ..

MODEL=AITL_20_4
MODEL_DIR=model/$MODEL
TEST_NAME=AITL
IMAGE_DIR=output/${TEST_NAME}/adv_image

CUDA_VISIBLE_DEVICES=$1 python predict.py \
    --output_dir=$MODEL_DIR \
    --AITL_batch_size=1 \
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
    --AITL_model_id=1 \
    --AITL_num_sample=1000 \
    --attack_max_epsilon=16 \
    --attack_image_height=299 \
    --attack_image_width=299 \
    --attack_image_resize=330 \
    --attack_model_list=1 \
    --attack_num_iter=10 \
    --attack_num_noise=1 \
    --attack_untarget=True \
    --attack_output_dir=$IMAGE_DIR

EVAL_MODEL='1_2_3_4_5_6_7_8_9_10_11_12_13'
DATASET="imagenet_valrs"
UNTARGET=True

CUDA_VISIBLE_DEVICES=$1 python evaluate.py \
        --dataset=${DATASET} \
        --evaluate_model=${EVAL_MODEL} \
        --untarget=${UNTARGET} \
        --output_dir="./output" \
        --exp_name="${TEST_NAME}" \
        --batch_size=50