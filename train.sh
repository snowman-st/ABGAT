#ASGCN
#python train.py --model_name asgcn \
				--dataset mams \
				--optimizer adam \
				--learning_rate 0.001 \
				--num_epoch 100\
				--batch_size 32
#ASGAT

python train.py --model_name asgat2 \
				--dataset rest14 \
				--optimizer adam \
				--learning_rate 0.001 \
				--dropout  0.4 \
				--num_epoch 100\
				--batch_size 32 \
				--nheads 4 \
				--device 3 \
				--logfile log\/

python train.py --model_name asgat2 \
				--dataset rest14 \
				--optimizer adam \
				--learning_rate 0.003 \
				--dropout  0.4 \
				--num_epoch 100\
				--batch_size 32 \
				--nheads 4 \
				--device 3 \
				--logfile log\/
#ASCNN
#python train.py --model_name ascnn \
				--dataset rest14 \
				--optimizer adam \
				--learning_rate 0.001 \
				--num_epoch 100\
				--batch_size 32
				
#LSTM
#python train.py --model_name LSTM \
				--dataset rest14 \
				--optimizer adam \
				--learning_rate 0.001 \
				--num_epoch 100\
				--batch_size 32