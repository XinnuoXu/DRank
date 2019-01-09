#coding=utf8

if __name__ == '__main__':
	import os
	import numpy as np
	import matplotlib.pyplot as plt

	# Plot length of dialogues
	dial_lens = []
	label_rate = 0.9
	for filename in os.listdir("./"):
		if filename.find("generated_dial_examples") == -1:
			continue
		for line in open(filename):
			dial_lens.append((len(line.strip().split("\t"))-1)/2)
	n, bins, patches = plt.hist(dial_lens, 10, density=False, facecolor='g', alpha=0.75)

	label_x = 0
	for i in range(0, len(n)):
		if sum(n[:i]) > len(dial_lens) * label_rate:
			label_x = bins[i]
			break
	plt.plot([label_x, label_x], [0, 1200], 'r--', linewidth=1)
	plt.xlabel('Fumber of dialogue turns')
	plt.ylabel('Percentage')
	plt.title('Histogram of dialogue turns (AMT)')
	plt.text(label_x+1, 900, r'90% dials are less than ' + str(label_x) + r' turns')
	plt.grid(True)
	plt.savefig("hist_dial_turns.png")
