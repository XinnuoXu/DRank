#coding=utf8
from collections import defaultdict
import numpy as np

def cal_entropy(generated):
	#print 'in BLEU score calculation'
	#the maximum is bigram, so assign the weight into 2 half.
	etp_score = [0.0,0.0,0.0,0.0]
	div_score = [0.0,0.0,0.0,0.0]
	counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
	for gg in generated:
		g = gg.rstrip('2').split()
		for n in range(4):
			for idx in range(len(g)-n):
				ngram = ' '.join(g[idx:idx+n+1])
				counter[n][ngram] += 1
	for n in range(4):
		total = sum(counter[n].values()) +1e-10
		for v in counter[n].values():
			etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
		div_score[n] = (len(counter[n].values())+0.0) /total
	return etp_score, div_score 

def diversity(line_list):
	uni_set = set(); uni_num = 0
	bi_set = set(); bi_num = 0
	for line in line_list:
		flist = line.split(" ")
		for x in flist:
			uni_set.add(x)
			uni_num += 1
		for i in range(0, len(flist)-1):
			bi_set.add(flist[i] + "<XXN>" + flist[i + 1])
			bi_num += 1
	d1 = len(uni_set) / float(uni_num)
	d2 = len(bi_set) / float(bi_num)
	print ("DIVERSE-1", d1)
	print ("DIVERSE-2", d2)
	print ("DISTINCT SENTENCES", len(set(line_list)) / float(len(line_list)))
	print ("ENTROPY", cal_entropy(line_list))

if __name__ == '__main__':
	import sys
	import random
	line_list = []
	turns = 0
	asked_turns = int(sys.argv[1])
	for line in sys.stdin:
		line = line.strip()
		if line == "":
			turns = 0
			continue
		if line.startswith("Reward"):
			continue
		if asked_turns == -1:
			if turns % 2 == 1:
				line_list.append(line)
		else:
			if turns == asked_turns:
				line_list.append(line)
		turns += 1
	diversity(line_list)
