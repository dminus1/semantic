# This is based on https://nlp.gluon.ai/examples/sentence_embedding/bert.html

import argparse
import random
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
from gluonnlp.model import bert_12_768_12
from tokenization import FullTokenizer
from dataset import MRPCDataset, ClassificationTransform
import os
import pdb
import time
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_fscore_support

__all__ = ['BERTClassifier']
from mxnet.gluon import Block
from mxnet.gluon import nn
num_classes = 6 #BLESS
class BERTClassifier(Block):
    """Model for sentence (pair) classification task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for
    classification.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    num_classes : int, default is 2
        The number of target classes.
    dropout : float or None, default 0.0.
        Dropout probability for the bert output.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """
    def __init__(self, bert, num_classes=num_classes, dropout=0.0, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes, flatten=False))

    def forward(self, inputs, token_types, valid_length=None): # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, num_classes)
        """
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler_out)

np.random.seed(0)
random.seed()
mx.random.seed(2)
logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size. Number of examples per gpu in a minibatch')
parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='report interval')
parser.add_argument('--max_len', type=int, default=128, help='Maximum length of the sentence pairs')
parser.add_argument('--gpu', action='store_true', help='whether to use gpu for finetuning')
args = parser.parse_args()
logging.info(args)
batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr

ctx = mx.cpu(0)

dataset = 'book_corpus_wiki_en_uncased'
bert, vocabulary = bert_12_768_12(dataset_name=dataset,
                                  pretrained=True, ctx=ctx, use_pooler=True,
                                  use_decoder=False, use_classifier=False)
do_lower_case = 'uncased' in dataset
tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)

model = BERTClassifier(bert, dropout=0.1)
model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
logging.info(model)
model.hybridize(static_alloc=True)

loss_function = gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()

trans = ClassificationTransform(tokenizer, MRPCDataset.get_labels(), args.max_len)
data_train = MRPCDataset('train').transform(trans)

bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=batch_size,
                                           shuffle=True, last_batch='rollover')

data_dev = MRPCDataset('dev').transform(trans)
bert_dataloader_dev = mx.gluon.data.DataLoader(data_dev, batch_size=test_batch_size,
                                               shuffle=False)

def evaluate():
    """Evaluate the model on validation dataset.
    """
    if not os.path.exists("glue_data/MRPC/dev.tsv"):
        print ("missing dev.tsv")
        time.sleep(60)
        return

    data_dev = MRPCDataset('dev').transform(trans)
    bert_dataloader_dev = mx.gluon.data.DataLoader(data_dev, batch_size=1,
                                                   shuffle=False)  #
    step_loss = 0
    metric1 = mx.metric.Accuracy()
    metric1.reset()
    f = open("scores-3.txt", "w")
    f1 = open("bert-results-3.txt", "w")
    count = 0
    flabs = open("bert-run-2." + str(random.randint(0, 10000)) + ".txt", "w")
    for _, seqs in enumerate(bert_dataloader_dev):
        Ls = []
        input_ids, valid_len, type_ids, label = seqs
        # '''
        out = model(input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                    valid_len.astype('float32').as_in_context(ctx))
        ls = loss_function(out, label.as_in_context(ctx)).mean()
        Ls.append(ls)
        step_loss += sum([L.asscalar() for L in Ls])
        # f.write(ls)
        metric1.update([label], [out])
        score = str(sum([L.asscalar() for L in Ls]))
        f.write(score + '\n')
        count += 1
        # if count > 10: #
        #    break
        winner = -1e10
        for i in range(num_classes):
            report = str(out[0][i]).split('\n')[1].replace('[', '').replace(']', '')
            if float(report) > winner:
                predicted = i
                winner = float(report)

        flabs.write(str(predicted))

        if count % 100 == 0:
            if (os.path.exists("stop.txt") == True):
                break

    logging.info('validation accuracy: %s', metric1.get()[1])
    logging.info('\tav cost: %s', step_loss / max(count, 1))
    pre, rec, f1_, support = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    logging.info('f1: %s', f1_)
    logging.info('pre: %s', pre)
    logging.info('rec: %s', rec)

    f1.write(str(metric1.get()[1]) + '\n')
    f.close()
    f1.close()
    flabs.close()


def train():

 differentiable_params = []

    # Do not apply weight decay on LayerNorm and bias terms
 for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

 for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)


 TrainSet = False
 while True:

    if not TrainSet or os.path.exists("update-train.txt"):
        print ("Loading training data")
        TrainSet = True
        """Training function."""
        trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                                {'learning_rate': lr, 'epsilon': 1e-9})

        num_train_examples = len(data_train)
        num_train_steps = int(num_train_examples / batch_size * args.epochs)
        warmup_ratio = args.warmup_ratio
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        step_num = 0


    eval = False
    print ("Training started")
    for epoch_id in range(10000):

        metric.reset()
        step_loss = 0
        for batch_id, seqs in enumerate(bert_dataloader):

            if os.path.exists("update-train.txt"):
                break

            while (os.path.exists("pause.txt") == True):
                pdb.set_trace()


            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer.set_learning_rate(max(1e-5, new_lr))
            with mx.autograd.record():
                input_ids, valid_length, type_ids, label = seqs
                out = model(input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                            valid_length.astype('float32').as_in_context(ctx))
                ls = loss_function(out, label.as_in_context(ctx)).mean()
            ls.backward()
            grads = [p.grad(ctx) for p in differentiable_params]
            gluon.utils.clip_global_norm(grads, 1)
            trainer.step(1)
            step_loss += ls.asscalar()
            metric.update([label], [out])
            all(label < num_classes)
            if (batch_id + 1) % (10) == 0:
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                             .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                     step_loss / args.log_interval,
                                     trainer.learning_rate, metric.get()[1]))
                step_loss = 0
                mx.nd.waitall()
                evaluate()

if __name__ == '__main__':
    time.sleep(1)
    train()

