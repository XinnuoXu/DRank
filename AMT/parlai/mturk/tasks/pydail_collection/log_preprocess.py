#coding=utf8

def write_file(dials, fpout):
	index = 0
	for dial in dials:
		tmp_dials = []
		try:
			for utt in dial:
				tokens = []
				for tok in generate_tokens(io.StringIO(utt.lower()).readline):
					_, t_str, _, _, _ = tok
					tokens.append(t_str)
				tmp_dials.append(" ".join(tokens))
		except:
			continue
		index += 1
		fpout.write("\t".join(tmp_dials) + "\n")
	fpout.close()
	return index

if __name__ == '__main__':
	import sys
	yes_good = 0; yes_bad = 0
	no = 0; invalid = 0
	dial = []; last_user = ""

	turn_num_yes_good = 0;
	turn_num_yes_bad = 0;
	turn_num_no = 0;

	fpout = open("AMT_preprocessed.txt", "w")
	pos_dials = []
	neg_dials = []
	for line in open("AMT_dials.txt"):
		line = line.strip()
		if line == "":
			score = -1
			tag = ""
			task_finishing = task_finishing.lower()
			if task_finishing.find("yes") > -1 or task_finishing == "y":
				if label == "0":
					tag = "pos"
					yes_good += 1
					score = 1.0
					turn_num_yes_good += turn_num
				else:
					tag = "neg"
					yes_bad += 1
					score = 1 - len(label.replace(" ", "").split(",")) / turn_num
					turn_num_yes_bad += turn_num
			elif task_finishing.find("no") > -1 or task_finishing == "n":
				tag = "neg"
				no += 1
				score = 0.0
				turn_num_no += turn_num
			else:
				tag = "inv"
				invalid += 1
			if tag == "pos":
				pos_dials.append(dial)
			if tag == "neg":
				neg_dials.append(dial)
			fpout.write("\n".join(dial) + "\n")
			fpout.write("task_finishing: " + task_finishing + "\n")
			fpout.write("label: " + label + "\n")
			fpout.write("turn_number: " + str(turn_num) + "\n")
			fpout.write("score: " + str(score) + "\n")
			fpout.write("tag: " + tag + "\n")
			fpout.write("\n")

			label = ""
			task_finishing = ""
			last_user = ""
			need = ""
			dial = []
			continue
		flist = line.split("\t")
		if flist[0] == "SYS" and flist[1].startswith("In this task"):
			yes_no_request = False
			continue
		if flist[0] == "SYS":
			if flist[1].startswith("[Turn "):
				if last_user != "":
					dial.append(last_user)
				sys_list = flist[1].split("] ")
				turn_num = int(sys_list[0].replace("[Turn ", ""))
				sys_str = sys_list[1]
				yes_no_request = False
				dial.append(sys_str)
			elif flist[1].startswith("Please tell me if you could find"):
				yes_no_request = True
			else:
				yes_no_request = False
				continue
		if flist[0] == "USER":
			if yes_no_request:
				task_finishing = flist[1]
			else:
				last_user = flist[1]
		if flist[0] == "LABEL":
			label = flist[1]

	# Statistic results
	print ("Avg length: ", (turn_num_yes_good + turn_num_yes_bad + turn_num_no)/(yes_good + yes_bad + no), "\n")
	print ("yes_good: ", yes_good, turn_num_yes_good/yes_good)
	print ("yes_bad: ", yes_bad, turn_num_yes_bad/yes_bad)
	print ("no: ", no, turn_num_no/no, "\n")
	print ("positive: ", yes_good, turn_num_yes_good/yes_good)
	print ("negative: ", yes_bad + no, (turn_num_no + turn_num_yes_bad)/(no + yes_bad), "\n")
	print ("invalid: ", invalid)
	fpout.close()


	# Seg train/dev/test
	import random
	import io
	from tokenize import generate_tokens
	random.shuffle(pos_dials)
	random.shuffle(neg_dials)
	# split train/dev/test = 7:1:2
	pos_train = pos_dials[: int(len(pos_dials) * 0.7)]
	pos_dev = pos_dials[int(len(pos_dials) * 0.7): int(len(pos_dials) * 0.8)]
	pos_test = pos_dials[int(len(pos_dials) * 0.8):]
	neg_train = neg_dials[: int(len(neg_dials) * 0.7)]
	neg_dev = neg_dials[int(len(neg_dials) * 0.7): int(len(neg_dials) * 0.8)]
	neg_test = neg_dials[int(len(neg_dials) * 0.8):]
	# Tokenize
	print ("train pos:", write_file(pos_train, open("generated_dial_examples_train.pos", "w")))
	print ("train neg:", write_file(neg_train, open("generated_dial_examples_train.neg", "w")))
	print ("dev pos:", write_file(pos_dev, open("generated_dial_examples_dev.pos", "w")))
	print ("dev neg:", write_file(neg_dev, open("generated_dial_examples_dev.neg", "w")))
	print ("test pos:", write_file(pos_test, open("generated_dial_examples_test.pos", "w")))
	print ("test neg:", write_file(neg_test, open("generated_dial_examples_test.neg", "w")))

