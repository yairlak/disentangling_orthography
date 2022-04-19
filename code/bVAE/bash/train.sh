BETAS='0.01 0.1 1 2 4 8'
BATCH_SIZES='64 128'
LATENT_SIZES='4 8 16 32 64 128'
LEARNING_RATES='0.001 0.01 0.05'

# NUMBER OF STIMULI

for BETA in $BETAS; do
    for BATCH_SIZE in $BATCH_SIZES; do
        for LATENT_SIZE in $LATENT_SIZES; do
            for LEARNING_RATE in $LEARNING_RATES; do
                LOG_PATH='logs/log_beta_'$BETA'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE'.pkl'
                cmd='python train.py --beta '$BETA' --batch-size '$BATCH_SIZE' --latent-size '$LATENT_SIZE' --learning-rate '$LEARNING_RATE' --log-path '$LOG_PATH' --use-cuda'
                echo $cmd
            done
        done
    done
done
