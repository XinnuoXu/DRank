#coding=utf8

import os, sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	x = []
	y_dict = []
	color_d = {'Sup500':'b', 'StepGAN+Sup500':'b', \
		'Sup1000':'g', 'StepGAN+Sup1000':'g', \
		'Sup2000':'r', 'StepGAN+Sup2000':'r'}
	for line in open("semi.txt"):
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
		if item[0].startswith("StepGAN"):
			plt.plot(x, item[1:], linewidth=1, label=item[0], color=color_d[item[0]])
		else:
			plt.plot(x, item[1:], linewidth=1, linestyle='--', label=item[0], color=color_d[item[0]])

	plt.xlabel('Evaluation metrics')
	plt.ylabel('Ranking performance (' + sys.argv[1] + '@k)')
	plt.title('Semi-supervised StepGAN vs. Supervised settings')
	#plt.text(label_x+1, 900, r'90% dials are less than ' + str(label_x) + r' turns')
	plt.grid(True)
	plt.legend()
	#plt.savefig("ndcg_acc.png")
	plt.savefig("semi.png")
