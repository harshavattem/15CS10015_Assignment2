import sys

if(len(sys.argv) != 2):
	print('Wrong number of arguments')
	sys.exit()

if(sys.argv[1] == '--train'):
	print('train')
	pass
elif(sys.argv[1] == '--test'):
	print('test')
	pass
else:
	print('Train or test?')