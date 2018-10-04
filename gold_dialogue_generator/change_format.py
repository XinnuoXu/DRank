#coding=utf8

if __name__ == '__main__':
	import sys
	dial = []; dials = []
	for line in sys.stdin:
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

	fpout = open("gold_dialogues.in", "w")
        for dial in dials:
		fpout.write("\t".join(dial) + "\n")
