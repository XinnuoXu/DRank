#coding=utf8

if __name__ == '__main__':
	import os, sys
	import numpy as np
	import matplotlib.pyplot as plt

	dial_lens = []
	label_rate = 0.9
	x = []
	y_dict = {}
	for line in open("sup.txt"):
		if line.strip() == "":
			continue
		flist = line.strip().split(" ")
		training_num = int(flist[0])
		label = flist[1]
		#if label.find("NDCG") == -1:
		if label.find(sys.argv[1]) == -1:
			continue
		value = float(flist[2])
		if training_num not in y_dict:
			y_dict[training_num] = []
		y_dict[training_num].append(value)
		label = label.replace(sys.argv[1], "")
		if label not in x:
			x.append(label)
	y = []
	label_list = []
	for training_num in y_dict:
		plt.plot(x, y_dict[training_num], linewidth=1, label="size = " + str(training_num))

	plt.xlabel('Evaluation metrics')
	plt.ylabel('Ranking performance (' + sys.argv[1] + '@k)')
	plt.title(sys.argv[1] + '@k of Supervised Settings with Different Training Size')
	#plt.text(label_x+1, 900, r'90% dials are less than ' + str(label_x) + r' turns')
	plt.grid(True)
	plt.legend()
	#plt.savefig("ndcg_acc.png")
	plt.savefig(sys.argv[1] + "_acc.png")
