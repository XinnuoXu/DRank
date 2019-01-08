#coding=utf8

if __name__ == '__main__':
	import os
	import sys
	evl_metrics = ["NDCG@10","NDCG@20", "NDCG@30", "NDCG@40", \
		"NDCG@50", "NDCG@100", "NDCG@150", "NDCG@200", \
		"recall@10","recall@20", "recall@30", "recall@40", \
		"recall@50", "recall@100", "recall@150", "recall@200",\
		"precision@10", "precision@20", "precision@30", "precision@40", \
		"precision@50", "precision@100", "precision@150", "precision@200", "ACC"]
	file_list = [filename for filename in os.listdir("./") if filename.startswith("dis.res")]

	sum_score = {}
	for ev in evl_metrics:
		sum_score[ev] = 0.0
		
	for filename in file_list:
		for line in open(filename):
			flist = line.strip().split(" ")
			label = flist[0]
			score = float(flist[1])
			sum_score[label] += score

	for ev in evl_metrics:
		print (ev, sum_score[ev]/len(file_list))
		
