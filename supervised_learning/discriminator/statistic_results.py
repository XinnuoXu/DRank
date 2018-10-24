#coding=utf8

if __name__ == '__main__':
	import os
	import numpy as np
	import matplotlib.pyplot as plt

	dial_lens = []
	label_rate = 0.9
	x_x = set()
	y_dict = {}
	for line in open("ndcg.txt"):
		flist = line.strip().split(" ")
		training_num = int(flist[0])
		label = flist[1]
		if label.find("NDCG") == -1:
			continue
		value = float(flist[2])
		if label not in y_dict:
			y_dict[label] = {}
		y_dict[label][training_num] = value
		x_x.add(training_num)
	x = list(sorted(x_x))
	y = []
	label_list = []
	for label in y_dict:
		label_list.append(label)
		y.append([item[1] for item in sorted(y_dict[label].items(), key=lambda d:d[0])])

	for y_val, label in zip(y, label_list):
		print (label)
		plt.plot(x, y_val, linewidth=1, label=label)
	plt.xlabel('Number of training data')
	plt.ylabel('Ranking performance (NDCG@k)')
	plt.title('NDCG@k of Supervised Learning with Different Training Size')
	#plt.text(label_x+1, 900, r'90% dials are less than ' + str(label_x) + r' turns')
	plt.grid(True)
	plt.legend()
	plt.savefig("cls_acc.png")
