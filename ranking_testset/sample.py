#coding=utf8

def acc():
	correct_num = 0; all_num = 0.0
	for line in open("test.res"):
		flist = line.strip().split(" ")
		if flist[1] == flist[2]:
			correct_num += 1
		all_num += 1
	return correct_num / all_num

def replace_quit(line):
	flist = line.split("\t")
	if flist[-2].find("quit") > -1:
		flist[-2] = "thanks for your help , goodbye ."
	return "\t".join(flist)

if __name__ == '__main__':
	import random
	import os
	sample_num = 200
	all_neg = [replace_quit(line.strip()) for line in open("generated_dial_examples_test.neg")]
	all_pos = [replace_quit(line.strip()) for line in open("generated_dial_examples_test.pos")]
	random.shuffle(all_neg)
	random.shuffle(all_pos)
	fp_neg = open("AMT_test.neg", "w")
	fp_pos = open("AMT_test.pos", "w")
	fp_neg.write("\n".join(all_neg[:sample_num]) + "\n")
	fp_pos.write("\n".join(all_pos[:sample_num]) + "\n")

