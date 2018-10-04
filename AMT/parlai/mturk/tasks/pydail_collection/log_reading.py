#coding=utf8

if __name__ == '__main__':
	import sys
	import re
	id_dict = {}
	pattern = re.compile('^[0-9, ]+$')
	for line in sys.stdin:
		line = line.strip()
		if line.find("mturk INFO Manager") == -1:
			continue
		if line.find("mturk INFO Manager sending") > -1:
			label = "SYS"
		else:
			label = "USER"
		flist = line.split("\', \'")
		conversation_id = ""
		text = ""
		for item in flist:
			if item.find("conversation_id") > -1:
				conversation_id = item.split("\': \'")[1]
			elif item.find("text\': ") > -1:
				text = item.split("text\': ")[1][1:].replace("<b>", "").replace("</b>", "").replace("\", \'type\': \'MESSAGE", "").replace("\", \'id\': \'Worker", "")
		if conversation_id not in id_dict:
			id_dict[conversation_id] = []
			id_dict[conversation_id].append(label + "\t" + text)
		else:
			if label + "\t" + text != id_dict[conversation_id][-1]:
				id_dict[conversation_id].append(label + "\t" + text)

	legal_dials = 0
	wrong_label_detect = "I am sorry but there is no place"
	end_signal = "Please tell me if you could find a restaurant"
	for item in id_dict:
		# illegal dialogues
		if pattern.match(id_dict[item][-1].split("\t")[1]) == None:
			continue
		end_turn = -1
		for i in range(0, len(id_dict[item])):
			if id_dict[item][i].find(end_signal) > -1:
				end_turn = i
				break
		# detect wrong labeled turns
		turns = id_dict[item][-1].split("\t")[1].split(",")
		turns = [t.strip() for t in turns if t.strip() != "1"]
		candidates = " ".join([sent for sent in id_dict[item] if sent.find(wrong_label_detect) > -1])
		filtered_turns = [t for t in turns if candidates.find("Turn " + t) == -1]
		print ("\n".join(id_dict[item][:end_turn + 3]))
		if len(filtered_turns) == 0:
			print ("LABEL\t0")
		else:
			print ("LABEL\t" + ",".join(filtered_turns))
		print ("")
		legal_dials += 1
	print ("LEGAL DIALS: ", str(legal_dials))
