BATCH_SIZES='128 256'
LATENT_SIZES='1 2 4 8 16 32 64 128'
LEARNING_RATES='0.001 0.01 0.1'

# NUMBER OF STIMULI

for BATCH_SIZE in $BATCH_SIZES; do
	for LATENT_SIZE in $LATENT_SIZES; do
		for LEARNING_RATE in $LEARNING_RATES; do
			LOG_PATH='log_'$BATCH_SIZE'_'$LATENT_SIZE'_'$LEARNING_RATE'.pkl'
			cmd='python train.py --batch-size '$BATCH_SIZE' --latent-size '$LATENT_SIZE' --learning-rate '$LEARNING_RATE' --log-path '$LOG_PATH
			echo $cmd
		done
	done
done
