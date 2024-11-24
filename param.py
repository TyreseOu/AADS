PAD = 0
UNK = 1
SOS = 2
EOS = 3

data_root = "data"

src_encoder_path = "source-encoder.pt"
src_classifier_path = "source-classifier.pt"

tgt_encoder_path = "target-encoder.pt"

model_root = "snapshots"
d_model_path = "critic.pt"

num_gpu = 1
manual_seed = None

c_learning_rate = 5e-5
d_learning_rate = 1e-4

n_vocab = 30522
hidden_size = 768
intermediate_size = 3072
embed_dim = 300
kernel_num = 20
kernel_sizes = [3, 4, 5]
pretrain = True
embed_freeze = True
class_num = 2
dropout = 0.1
num_labels = 2
d_hidden_dims = 384
d_output_dims = 2
