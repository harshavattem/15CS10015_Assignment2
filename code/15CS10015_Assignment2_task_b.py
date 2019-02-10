
# coding: utf-8

# In[1]:


import sys

if(len(sys.argv) != 2):
    print('Wrong number of arguments')
    sys.exit()

import os
import numpy as np
import data_loader
import module
from data_loader import DataLoader

import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import nd, autograd, gluon

ctx = mx.cpu()
data_ctx = ctx
model_ctx = ctx


# # Custom Dataset class

# In[2]:


dl = DataLoader()

class CustomDataset:
    
    def __init__(self, mode, dataset = 'all'):
        self.x, self.y = dl.load_data(mode, dataset)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return len(self.y)   



    # In[4]:


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
     

if(sys.argv[1] == '--train'):


    # In[3]:


    batch_size = 64

    train_data = mx.gluon.data.DataLoader(CustomDataset('train', 'train'), batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(CustomDataset('train', 'validation'), batch_size, shuffle=False)

    epochs = 10
    num_examples = len(train_data)



    # ## Vanilla Network

    # In[5]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    net.collect_params().initialize(mx.init.Uniform(.1), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_vanilla = loss_arr.copy()

    filename = os.path.join('../weights','b_vanilla.params')
    net.save_parameters(filename)


    # # Initialization

    # ## Normal Initialization

    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_norm_init = loss_arr.copy()

    filename = os.path.join('../weights','b_normal.params')
    net.save_parameters(filename)


    # ## Xavier Initialization

    # In[ ]:


    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    
        
    loss_xavier_init = loss_arr.copy()

    filename = os.path.join('../weights','b_xavier.params')
    net.save_parameters(filename)


    # ## Orthogonal Initialization

    # In[ ]:


    net.collect_params().initialize(mx.init.Orthogonal(scale=1.414, rand_type='uniform'), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))   
        
    loss_ortho_init = loss_arr.copy()

    filename = os.path.join('../weights','b_ortho.params')
    net.save_parameters(filename)


    # # Normalization

    # ## Batch Normalization

    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_batch_norm = loss_arr.copy()

    filename = os.path.join('../weights','b_batch.params')
    net.save_parameters(filename)


    # # Dropout

    # ## Dropout = 0.1

    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_dropout1 = loss_arr.copy()

    filename = os.path.join('../weights','b_dropout1.params')
    net.save_parameters(filename)


    # ## Dropout = 0.4

    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dropout(.4))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dropout(.4))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dropout(.4))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_dropout4 = loss_arr.copy()

    filename = os.path.join('../weights','b_dropout4.params')
    net.save_parameters(filename)


    # ## Dropout = 0.6

    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dropout(.6))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dropout(.6))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dropout(.6))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_dropout6 = loss_arr.copy()

    filename = os.path.join('../weights','b_dropout6.params')
    net.save_parameters(filename)


    # # Optimizers

    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))


    # ## Stochastic Gradient Descent

    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


    # In[ ]:


    def sgd(params, lr, batch_size):
        for param in params:
            param[:] = param - lr * param.grad / batch_size


    # In[ ]:


    loss_arr = []
    lr = .0005
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            w = []
            for i in range(3):
                w.append(net[i].weight.data())
                w.append(net[i].bias.data())
            sgd(w, lr, data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    
        
    loss_sgd = loss_arr.copy()

    filename = os.path.join('../weights','b_sgd.params')
    net.save_parameters(filename)


    # ## Nesterov’s accelerated momentum

    # In[6]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),'nag', {'momentum':.1, 'learning_rate':lr})


    # In[7]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            w = []
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
       
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print(cumulative_loss)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    
        
    loss_nest_opt = loss_arr.copy()

    filename = os.path.join('../weights','b_nag.params')
    net.save_parameters(filename)


    # ## AdaDelta

    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),'adadelta')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_adadelta_opt = loss_arr.copy()

    filename = os.path.join('../weights','b_adadelta.params')
    net.save_parameters(filename)


    # ## AdaGrad

    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adagrad')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))   
        
    loss_adagrad_opt = loss_arr.copy()

    filename = os.path.join('../weights','b_adagrad.params')
    net.save_parameters(filename)


    # ## RMSProp

    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'rmsprop')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    

    loss_rmsprop_opt = loss_arr.copy()

    filename = os.path.join('../weights','b_rmsprop.params')
    net.save_parameters(filename)


    # ## Adam

    # In[ ]:


    net.collect_params().initialize(mx.init.Normal(sigma=.05), ctx=model_ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')


    # In[ ]:


    loss_arr = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            
            with autograd.record():
                data = mx.ndarray.cast(data, dtype='float32')
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        
        loss_arr.append(cumulative_loss/num_examples)

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))    
        
    loss_adam_opt = loss_arr.copy()

    filename = os.path.join('../weights','b_adam.params')
    net.save_parameters(filename)


    # In[ ]:


    # plt.figure(figsize=(15,15))
    plt.plot(loss_vanilla, label='Vanilla')
    plt.plot(loss_norm_init, label='Normal Initialization')
    plt.plot(loss_xavier_init, label='Xavier Initialization')
    plt.plot(loss_ortho_init, label='Orthogonal Initialization')
    plt.legend(fontsize=15)
    plt.title('Initialization Methods Comparison')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.show()


    # In[ ]:


    # plt.figure(figsize=(15,15))
    plt.plot(loss_vanilla, label='Vanilla')
    plt.plot(loss_batch_norm, label='Batch Normalization')
    plt.legend(fontsize=15)
    plt.title('Normalization Methods Comparison')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.show()


    # In[ ]:


    # plt.figure(figsize=(15,15))
    plt.plot(loss_vanilla, label='Vanilla')
    plt.plot(loss_dropout1, label='Dropout(0.1)')
    plt.plot(loss_dropout4, label='Dropout(0.4)')
    plt.plot(loss_dropout6, label='Dropout(0.6)')
    plt.legend(fontsize=15)
    plt.title('Dropout Comparison')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.show()


    # In[ ]:


    # plt.figure(figsize=(15,15))
    plt.plot(loss_vanilla, label='Vanilla')
    plt.plot(loss_sgd, label='SGD Optimization')
    plt.plot(loss_nest_opt, label='Nesterov\'s Optimization')
    plt.plot(loss_adadelta_opt, label='AdaDelta Optimization')
    plt.plot(loss_adagrad_opt, label='AdaGrad Optimization')
    plt.plot(loss_rmsprop_opt, label='RMSProp Optimization')
    plt.plot(loss_adam_opt, label='Adam Optimization')
    plt.legend(fontsize=15)
    plt.title('Optimization Methods Comparison')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.show()

elif(sys.argv[1] == '--test'):


    # # TESTING

    batch_size = 64
    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))
    test_data = mx.gluon.data.DataLoader(CustomDataset('test'), batch_size, last_batch='keep', shuffle=False)


    # In[ ]:


    filename = os.path.join('../weights','b_vanilla.params')
    if not (os.path.isfile(filename)):
        print('No data for Vanilla')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Vanilla) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    filename = os.path.join('../weights','b_normal.params')
    if not (os.path.isfile(filename)):
        print('No data for Normal Initialization')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Normal Initialization) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    filename = os.path.join('../weights','b_xavier.params')
    if not (os.path.isfile(filename)):
        print('No data for Xavier Initialization')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Xavier Initialization) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    filename = os.path.join('../weights','b_ortho.params')
    if not (os.path.isfile(filename)):
        print('No data for Orthogonal Initialization')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Orthogonal Initialization) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    filename = os.path.join('../weights','b_batch.params')
    if not (os.path.isfile(filename)):
        print('No data for Batch Normalization')
    else:
        net.load_parameters(filename, ctx=ctx, allow_missing=True, ignore_extra=True)
        print('Accuracy(Batch Normalization) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    filename = os.path.join('../weights','b_dropout1.params')
    if not (os.path.isfile(filename)):
        print('No data for Dropout(0.1)')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Dropout(0.1)) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dropout(.4))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dropout(.4))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dropout(.4))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    filename = os.path.join('../weights','b_dropout4.params')
    if not (os.path.isfile(filename)):
        print('No data for Dropout(0.4)')
    else:
        net.load_parameters(filename, ctx=ctx, allow_missing=True)
        print('Accuracy(Dropout(0.4)) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dropout(.6))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dropout(.6))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dropout(.6))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    filename = os.path.join('../weights','b_dropout6.params')
    if not (os.path.isfile(filename)):
        print('No data for Dropout(0.6)')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Dropout(0.6)) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[9]:


    layer = [1024, 512, 256]
    lout = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(layer[0], activation="relu"))
        net.add(gluon.nn.Dense(layer[1], activation="relu"))
        net.add(gluon.nn.Dense(layer[2], activation="relu"))
        net.add(gluon.nn.Dense(lout))


    # In[ ]:


    filename = os.path.join('../weights','b_sgd.params')
    if not (os.path.isfile(filename)):
        print('No data for SGD')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(SGD) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[10]:


    filename = os.path.join('../weights','b_nag.params')
    if not (os.path.isfile(filename)):
        print('No data for Nesterov\'s Accelerated Momentum')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Nesterov\'s Accelerated Momentum) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    filename = os.path.join('../weights','b_adadelta.params')
    if not (os.path.isfile(filename)):
        print('No data for AdaDelta')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(AdaDelta) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    filename = os.path.join('../weights','b_adagrad.params')
    if not (os.path.isfile(filename)):
        print('No data for AdaGrad')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(AdaGrad) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    filename = os.path.join('../weights','b_rmsprop.params')
    if not (os.path.isfile(filename)):
        print('No data for RMSProp')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(RMSProp) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


    # In[ ]:


    filename = os.path.join('../weights','b_adam.params')
    if not (os.path.isfile(filename)):
        print('No data for Adam')
    else:
        net.load_parameters(filename, ctx=ctx)
        print('Accuracy(Adam) = ' + str(100 * evaluate_accuracy(test_data, net)) + '%')


else:
    print('Train or test?')
