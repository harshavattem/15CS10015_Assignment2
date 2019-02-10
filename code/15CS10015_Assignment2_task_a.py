
import sys

if(len(sys.argv) != 2):
	print('Wrong number of arguments')
	sys.exit()

import numpy as np
import data_loader
import module
from data_loader import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

import mxnet as mx
from mxnet import nd, autograd, gluon



ctx = mx.cpu()
data_ctx = ctx
model_ctx = ctx


# ## Custom Dataset class


dl = DataLoader()

class CustomDataset:
	
	def __init__(self, mode, dataset = 'all'):
		self.x, self.y = dl.load_data(mode, dataset)
	
	def __getitem__(self, i):
		return self.x[i], self.y[i]
	
	def __len__(self):
		return len(self.y)        

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

	# # TRAINING

	batch_size = 64

	train_data = mx.gluon.data.DataLoader(CustomDataset('train', 'train'), batch_size, shuffle=True)
	test_data = mx.gluon.data.DataLoader(CustomDataset('train', 'validation'), batch_size, shuffle=False)

	epochs = 10
	num_examples = len(train_data)


	# ## Deep Network

	layer = [512, 128, 64, 32, 16]
	lout = 10
	net1 = gluon.nn.Sequential()
	with net1.name_scope():
		net1.add(gluon.nn.Dense(layer[0], activation="relu"))
		net1.add(gluon.nn.Dense(layer[1], activation="relu"))
		net1.add(gluon.nn.Dense(layer[2], activation="relu"))
		net1.add(gluon.nn.Dense(layer[3], activation="relu"))
		net1.add(gluon.nn.Dense(layer[4], activation="relu"))
		net1.add(gluon.nn.Dense(lout))


	net1.collect_params().initialize(mx.init.Uniform(.1), ctx=model_ctx, force_reinit=True)
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net1.collect_params(), 'adam', {'learning_rate': .001})


	loss_arr = []
	valid_acc = []
	for e in range(epochs):
		cumulative_loss = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			
			with autograd.record():
				data = mx.ndarray.cast(data, dtype='float32')
				output = net1(data)
				loss = softmax_cross_entropy(output, label)
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_loss += nd.sum(loss).asscalar()
		
		loss_arr.append(cumulative_loss/num_examples)

		test_accuracy = evaluate_accuracy(test_data, net1)
		train_accuracy = evaluate_accuracy(train_data, net1)
		print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %(e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
		valid_acc.append(test_accuracy)

	loss_arr1 = loss_arr.copy()
	valid_acc1 = valid_acc.copy()


	filename = os.path.join('weights','a1.params')
	net1.save_parameters(filename)


	# ## Shallow Network

	layer = [1024, 512, 256]
	lout = 10
	net2 = gluon.nn.Sequential()
	with net2.name_scope():
		net2.add(gluon.nn.Dense(layer[0], activation="relu"))
		net2.add(gluon.nn.Dense(layer[1], activation="relu"))
		net2.add(gluon.nn.Dense(layer[2], activation="relu"))
		net2.add(gluon.nn.Dense(lout))


	net2.collect_params().initialize(mx.init.Uniform(.1), ctx=model_ctx, force_reinit=True)
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net2.collect_params(), 'adam', {'learning_rate': .001})

	loss_arr = []
	valid_acc = []
	for e in range(epochs):
		cumulative_loss = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			
			with autograd.record():
				data = mx.ndarray.cast(data, dtype='float32')
				output = net2(data)
				loss = softmax_cross_entropy(output, label)
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_loss += nd.sum(loss).asscalar()
		
		loss_arr.append(cumulative_loss/num_examples)

		test_accuracy = evaluate_accuracy(test_data, net2)
		train_accuracy = evaluate_accuracy(train_data, net2)
		print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %(e, cumulative_loss/num_examples, train_accuracy, test_accuracy)) 
		valid_acc.append(test_accuracy)

	loss_arr2 = loss_arr.copy()
	valid_acc2 = valid_acc.copy()


	filename = os.path.join('weights','a2.params')
	net2.save_parameters(filename)


	# ## Graphical Comparison

	plt.figure(figsize=(8,6))
	plt.xlabel('Epochs', fontsize=15)
	plt.ylabel('Loss', fontsize=15)
	plt.plot(loss_arr1, label='Deep Network', linewidth=3)
	plt.plot(loss_arr2, label = 'Shallow Network', linewidth=3)
	plt.legend(fontsize='x-large')
	plt.show()


	plt.figure(figsize=(8,6))
	plt.xlabel('Epochs', fontsize=15)
	plt.ylabel('Validation', fontsize=15)
	plt.plot(valid_acc1, label='Deep Network', linewidth=3)
	plt.plot(valid_acc2, label = 'Shallow Network', linewidth=3)
	plt.legend(fontsize='x-large')
	plt.show()

elif(sys.argv[1] == '--test'):

	# # TESTING

	test_data = mx.gluon.data.DataLoader(CustomDataset('test'), 64, last_batch='keep', shuffle=False)

	layer = [512, 128, 64, 32, 16]
	lout = 10
	net1 = gluon.nn.Sequential()
	with net1.name_scope():
		net1.add(gluon.nn.Dense(layer[0], activation="relu"))
		net1.add(gluon.nn.Dense(layer[1], activation="relu"))
		net1.add(gluon.nn.Dense(layer[2], activation="relu"))
		net1.add(gluon.nn.Dense(layer[3], activation="relu"))
		net1.add(gluon.nn.Dense(layer[4], activation="relu"))
		net1.add(gluon.nn.Dense(lout))
	filename = os.path.join('weights','a1.params')
	if not (os.path.isfile(filename)):
		print('No data for NN1')
	else:
		net1.load_parameters(filename, ctx=ctx)
		print('Accuracy(Deep Network) = ' + str(100 * evaluate_accuracy(test_data, net1)) + '%')

	layer = [1024, 512, 256]
	lout = 10
	net2 = gluon.nn.Sequential()
	with net2.name_scope():
		net2.add(gluon.nn.Dense(layer[0], activation="relu"))
		net2.add(gluon.nn.Dense(layer[1], activation="relu"))
		net2.add(gluon.nn.Dense(layer[2], activation="relu"))
		net2.add(gluon.nn.Dense(lout))
	filename = os.path.join('weights','a2.params')
	if not (os.path.isfile(filename)):
		print('No data for NN2')
	else:
		net2.load_parameters(filename, ctx=ctx)
		print('Accuracy(Shallow Network) = ' + str(100 * evaluate_accuracy(test_data, net2)) + '%')

else:
	print('Train or test?')