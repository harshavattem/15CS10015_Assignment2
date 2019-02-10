
# coding: utf-8

# In[ ]:


import sys

if(len(sys.argv) != 2):
    print('Wrong number of arguments')
    sys.exit()


import os
import numpy as np
import data_loader
import module
from data_loader import DataLoader

from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.externals import joblib

import mxnet as mx
from mxnet import nd, autograd, gluon

ctx = mx.cpu()
data_ctx = ctx
model_ctx = ctx


# In[ ]:


dl = DataLoader()

class CustomDataset:
    
    def __init__(self, mode, dataset = 'all'):
        self.x, self.y = dl.load_data(mode, dataset)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return len(self.y)        


# In[ ]:


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        data = mx.ndarray.cast(data, dtype='float32')
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# ## TRAINING

if(sys.argv[1] == '--train'):


    train_set = CustomDataset('train', 'train')
    valid_set = CustomDataset('train', 'validation')


    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    filename = os.path.join('../weights','a2.params')
    if not (os.path.isfile(filename)):
        print('No data')
        sys.exit()
    else:
        net.load_parameters(filename, ctx=ctx)


    # In[ ]:


    for i in range(0,3):
        if i == 0:
        	train_x = nd.relu(net[0](mx.nd.array(train_set.x).reshape(len(train_set),784)))
        	valid_x = nd.relu(net[0](mx.nd.array(valid_set.x).reshape(len(valid_set),784)))
        else :
        	train_x = nd.relu(net[i](train_x))
        	valid_x = nd.relu(net[i](valid_x))
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_x.asnumpy(), train_set.y)
        print('Layer ' + str(i) + ':')
        print('Train accuracy: ' + str(100 * clf.score(train_x.asnumpy(), train_set.y)) + '%')
        print('Validation accuracy: ' + str(100 * clf.score(valid_x.asnumpy(), valid_set.y)) + '%')
        print()
        
        layername = str(i)
        filename = os.path.join('../weights','c' + layername + '.joblib')
        joblib.dump(clf, filename)
        

elif(sys.argv[1] == '--test'):

    # ## TESTING

    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    filename = os.path.join('../weights','a2.params')
    if not (os.path.isfile(filename)):
        print('No data')
        sys.exit()
    else:
        net.load_parameters(filename, ctx=ctx)


    # In[ ]:


    test_set = CustomDataset('test', '')


    # In[ ]:


    for i in range(0,3):
        if i == 0:
        	test_x = nd.relu(net[0](mx.nd.array(test_set.x).reshape(len(test_set),784)))
        else :
        	test_x = nd.relu(net[i](test_x))
            
        layername = str(i)
        filename = os.path.join('../weights','c' + layername + '.joblib')
        clf = joblib.load(filename)
            
        print('Accuracy (Layer '+ str(i) + '): ' + str(100 * clf.score(test_x.asnumpy(), test_set.y)) + '%')


else:
    print('Train or test?')
