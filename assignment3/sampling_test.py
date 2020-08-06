import pickle

import matplotlib.pyplot as plt

from cs231n.coco_utils import sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions

small_rnn_model = pickle.load(open('saved_models/small_rnn', 'rb'))
data = load_coco_data(pca_features=True)
small_data = load_coco_data(max_train=50)

for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = small_rnn_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.figure()
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()
