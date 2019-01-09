#coding=utf8

import os, sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	x = []
	y_dict = []
	for line in open("gans.txt"):
		flist = line.strip().split("\t")
		if len(flist) == 4:
			for i in range(0, 4):
				y_dict.append([flist[i]])
			continue
		x.append(flist[0].replace(sys.argv[1], ""))
		for i in range(1, 5):
			y_dict[i-1].append(float(flist[i]))
	y = []
	label_list = []
	for item in y_dict:
		plt.plot(x, item[1:], linewidth=1, label=item[0])

	plt.xlabel('Evaluation metrics')
	plt.ylabel('Ranking performance (' + sys.argv[1] + '@k)')
	plt.title('StepGAN with Different Training Size vs. Supervised Approach')
	#plt.text(label_x+1, 900, r'90% dials are less than ' + str(label_x) + r' turns')
	plt.grid(True)
	plt.legend()
	#plt.savefig("ndcg_acc.png")
	plt.savefig("gans.png")
