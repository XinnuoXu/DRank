#coding=utf8

if __name__ == '__main__':
	import os
	import numpy as np
	import matplotlib.pyplot as plt

	x = []
	y = []
	for line in open("acc.txt"):
		flist = line.strip().split("\t")
		x.append(int(flist[0]))
		y.append(float(flist[1]))

	plt.plot(x, y, linewidth=1)
	plt.xlabel('Number of training data')
	plt.ylabel('Classification performance')
	plt.title('Classification ACC of Supervised Learning with Different Training Size')
	#plt.text(label_x+1, 900, r'90% dials are less than ' + str(label_x) + r' turns')
	plt.grid(True)
	plt.savefig("cls_acc.png")
