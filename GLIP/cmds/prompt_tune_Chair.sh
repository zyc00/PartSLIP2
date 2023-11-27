python -m torch.distributed.launch --nproc_per_node=1 tools/finetune.py \
    --config-file configs/pretrain/glip_Swin_L.yaml \
    --ft-tasks configs/partnetM/Chair.yaml \
    --skip-test \
    --custom_shot_and_epoch_and_general_copy 80_200_1 \
    --evaluate_only_best_on_test --push_both_val_and_test \
    MODEL.WEIGHT ./models/glip_large_model.pth \
    SOLVER.USE_AMP True \
    TEST.DURING_TRAINING True \
    SOLVER.IMS_PER_BATCH 3 \
    SOLVER.WEIGHT_DECAY 0.25 \
    TEST.EVAL_TASK detection \
    DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
    MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 \
    MODEL.DYHEAD.USE_CHECKPOINT True \
    SOLVER.TEST_WITH_INFERENCE True \
    SOLVER.USE_AUTOSTEP True \
    DATASETS.USE_OVERRIDE_CATEGORY True \
    SOLVER.SEED 10 \
    DATASETS.SHUFFLE_SEED 3 \
    DATASETS.USE_CAPTION_PROMPT False \
    DATASETS.DISABLE_SHUFFLE True \
    SOLVER.STEP_PATIENCE 3 \
    SOLVER.CHECKPOINT_PER_EPOCH 1.0 \
    SOLVER.AUTO_TERMINATE_PATIENCE 8 \
    SOLVER.MODEL_EMA 0.0 \
    SOLVER.TUNING_HIGHLEVEL_OVERRIDE full \
    SOLVER.BASE_LR 0.05 \
    SOLVER.TUNING_HIGHLEVEL_OVERRIDE language_prompt_v2 \
    TEST.IMS_PER_BATCH 3 \
    SOLVER.FIND_UNUSED_PARAMETERS False 