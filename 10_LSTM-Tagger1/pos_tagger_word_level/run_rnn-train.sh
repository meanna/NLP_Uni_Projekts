#small parameters
python rnn-train.py ../Tiger-morph/train.tagged ../Tiger-morph/dev.tagged param --num_epochs=2 --num_words=2000 --emb_size=30 --rnn_size=30 --dropout_rate=0.5 --learning_rate=0.001
