#small parameters
python rnn-train.py ../Tiger-morph/train.tagged ../Tiger-morph/dev.tagged --parfile param --num_epochs=2 --emb_size=20 --rnn_size=30 --dropout_rate=0.5 --learning_rate=0.001

#suggested parameters
#python rnn-train.py ../Tiger-morph/train.tagged ../Tiger-morph/dev.tagged --parfile param_reduced_lr --num_epochs=20 --emb_size=200 --rnn_size=300 --dropout_rate=0.5 --learning_rate=0.0001
