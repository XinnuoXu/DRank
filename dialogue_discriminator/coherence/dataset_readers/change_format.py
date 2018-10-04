#coding=utf8

if __name__ == '__main__':
	import sys
	dial = []; dials = []
	pos_train_fp = open("./generated_dial_examples_train.pos", "w")
	pos_dev_fp = open("./generated_dial_examples_dev.pos", "w")
	pos_test_fp = open("./generated_dial_examples_test.pos", "w")
	for line in open("/home/t-xinxu/pydial/generated_dial_examples.pos"):
		line = line.strip()
		flist = line.split("\t")
		if len(flist) == 1 and len(dial)!=0:
			dials.append(dial)
			dial = []
			continue
		if len(flist) == 2:
			if flist[1].find("NEW DIALOGUE") == -1:
				dial.append(flist[1].replace("USER:", "").replace("SYSTEM:", "").strip())
	if len(dial) != 0:
		dials.append(dial)
	readline = 0
	while readline < len(dials) * 0.6:
		pos_train_fp.write("\t".join(dials[readline]) + "\n")
		readline += 1
	while readline < len(dials) * 0.8:
		pos_dev_fp.write("\t".join(dials[readline]) + "\n")
		readline += 1
	while readline < len(dials):
		pos_test_fp.write("\t".join(dials[readline]) + "\n")
		readline += 1

	dial = []; del dials[:]
	neg_train_fp = open("./generated_dial_examples_train.neg", "w")
	neg_dev_fp = open("./generated_dial_examples_dev.neg", "w")
	neg_test_fp = open("./generated_dial_examples_test.neg", "w")
	for line in open("/home/t-xinxu/pydial/generated_dial_examples.neg"):
		line = line.strip()
		flist = line.split("\t")
		if len(flist) == 1 and len(dial)!=0:
			dials.append(dial)
			dial = []
			continue
		if len(flist) == 2:
			if flist[1].find("NEW DIALOGUE") == -1:
				dial.append(flist[1].replace("USER:", "").replace("SYSTEM:", "").strip())
	if len(dial) != 0:
		dials.append(dial)
	readline = 0
	while readline < len(dials) * 0.6:
		neg_train_fp.write("\t".join(dials[readline]) + "\n")
		readline += 1
	while readline < len(dials) * 0.8:
		neg_dev_fp.write("\t".join(dials[readline]) + "\n")
		readline += 1
	while readline < len(dials):
		neg_test_fp.write("\t".join(dials[readline]) + "\n")
		readline += 1
