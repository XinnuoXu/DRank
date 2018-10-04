#coding=utf8

if __name__ == '__main__':
	file_name = "ontology/ontologies/CamRestaurants-rules.json"
	lines = [line.strip().lower() for line in open(file_name)]
	import json
	dict_json = json.loads("".join(lines))
	dictionary = []
	for key in dict_json:
		if type(dict_json[key]) == list:
			dictionary.extend(dict_json[key])
		if type(dict_json[key]) == dict:
			for k in dict_json[key]:
				if type(dict_json[key][k]) == list:
					dictionary.extend(dict_json[key][k])
	dict_term = []
	for item in dictionary:
		dict_term.extend(item.split(" "))

	open("examples.dict", "w").write("\n".join(set(dict_term)) + "\n")
