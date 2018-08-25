import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SortNet(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(SortNet, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.fc1 = nn.Linear(input_size, hidden_size, bias = None)
		self.fc2 = nn.Linear(hidden_size, input_size)

	def forward(self, x):
		x = F.tanh(self.fc1(x))
		x = self.fc2(x)
		return x

def getxy(batch_size, count, max_val):
	x = np.random.randint(-max_val, max_val + 1, size = (batch_size, count))
	y = np.array([[sorted(arr1d).index(elm) for elm in arr1d] for arr1d in x])
	#y = np.argsort(x)

	x = torch.from_numpy(x).float()
	y = torch.from_numpy(y).float()
	return x, y
	

def train(batch_size, count, max_val, save_path, resume_flag = False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	input_size = count
	hidden_size = sum(range(input_size))	

	model = SortNet(input_size, hidden_size).to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adamax(model.parameters())

	def checkpoint(model, epoch, chk_path = 'sort_chk.pth'):
		torch.save(model.state_dict(), chk_path)


	print (model)
	print ('Model built successfully...')
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	print('Total params: {}'.format(sum([np.prod(p.size()) for p in model_parameters])))
	
	train_steps = 800
	val_steps = 200
	epochs = 1000

	if resume_flag:
		model.load_state_dict(torch.load(save_path))

	for epoch in range(epochs):
		train_loss = 0
		for batch_idx in range(train_steps):
			x, y = getxy(batch_size, count, max_val)
			pred = model(x)
			loss = criterion(pred, y)

			optimizer.zero_grad()
			train_loss += loss.item()
			loss.backward()		
			optimizer.step()

		print ("===> Epoch {} Complete: Avg. Training Loss: {:.4f}".format(epoch, 
										   train_loss / train_steps))
		val_loss = 0
		with torch.no_grad():
			for batch_idx in range(val_steps):
				x, y = getxy(batch_size, count, max_val)
				pred = model(x)
				loss = criterion(pred, y)

				val_loss += loss.item()
		print ("===> Epoch {} Complete: Avg. validation Loss: {:.4f}".format(epoch, 
									  	     val_loss / val_steps))

		checkpoint(model, epoch, save_path)

def test(batch_size, count, max_val, save_path):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	input_size = count
	hidden_size = sum(range(input_size))

	model = SortNet(input_size, hidden_size).to(device)
	model.load_state_dict(torch.load(save_path))

	x, y = getxy(batch_size, count, max_val)
	pred = model(x)
	for inp, tgt, out in zip(np.array(x), np.array(y), np.array(pred.detach())):
		out = [np.clip(round(idx), 0, count - 1) for idx in out]

		tgt_arr = np.zeros_like(inp)
		out_arr = np.zeros_like(inp)
		for elm, tgt_idx, out_idx in zip(inp, tgt, out):
			tgt_arr[int(tgt_idx)] = elm
			out_arr[int(out_idx)] = elm
		print ('Input array: {}'.format(inp))
		print ('Target array: {}'.format(tgt_arr))
		print ('Output array: {}'.format(out_arr))
		print('\n')
	
	
if __name__  == '__main__':
	parser = argparse.ArgumentParser(description = 'SortNet Parameters')
	
	parser.add_argument('-m',
			    '--exec_mode',
			    default = 'train',
			    help = 'Execution mode',
			    choices = ['train', 'test'])
	parser.add_argument('-b',
			    '--batch_size',
			    default = 32)
	parser.add_argument('-c',
			    '--count',
			    default = 12)
	parser.add_argument('-v',
			    '--max_val',
			    default = 10000)
	parser.add_argument('-s',
			    '--save_path',
			    default = 'sort_chk.pth')

	arguments = parser.parse_args()
	mode = arguments.exec_mode
	batch_size = arguments.batch_size
	count = arguments.count
	max_val = arguments.max_val
	save_path = arguments.save_path

	if mode == 'train':
		train(batch_size, count, max_val, save_path, resume_flag = False)
	else:
		test(batch_size, count, max_val, save_path)
	
