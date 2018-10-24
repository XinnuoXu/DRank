# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld

import Texthub
import argparse, re, os, sys, random
from Agent import DialogueAgent
from utils import ContextLogger
from utils import Settings
from ontology import Ontology

__author__ = "cued_dialogue_systems_group"
__version__ = Settings.__version__

global seed
#seed = Settings.init("/home/t-xinxu/ParlAI/parlai/mturk/tasks/pydail_collection/config/Tut-hdc-CamInfo.cfg", None)
seed = Settings.init("./config/Tut-hdc-CamInfo.cfg", None)
global Ontology
Ontology.init_global_ontology()

class QADataCollectionOnboardWorld(MTurkOnboardWorld):
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = 'Welcome onboard!'
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class QADataCollectionWorld(MTurkTaskWorld):
    """
    World for recording a turker's question and answer given a context.
    Assumes the context is a random context from a given task, e.g.
    from SQuAD, CBT, etc.
    """

    collector_agent_id = 'Restaurant bot'
    
    def __init__(self, opt, mturk_agent):
        self.mturk_agent = mturk_agent
        self.episodeDone = False
        self.with_res, self.without_res, self.inform_list = self.load_patterns()
        self.keywords = []
        self.wrongturns = []
        self.pattern = re.compile('^[0-9, ]+$')

    def load_patterns(self):
        with_res = []
        without_res = []
        #Price_range, Food_type, Area
        for line in open("/home/t-xinxu/ParlAI/parlai/mturk/tasks/pydail_collection/exam_ontology.res"):
            flist = line.strip().split("\t")
            if flist[0] == "0":
                without_res.append(flist[1:4])
            elif flist[0] == "1":
                with_res.append(flist[1:4])
        inform_list = ["phone number", "address"]
        return with_res, without_res, inform_list

    def task_generation_v1(self):
        task_patt = "In this task, you will need to talk with our restaurants searching bot. You want to find a <PRICE> <TYPE> restaurant <AREA>"
        task_2nd_patt = ". However, you may not find a restaurant on these conditions. You then change your mind to find a <PRICE> <TYPE> restaurant <AREA>"
        slot_num = random.randint(0, 4)
        if slot_num == 0:
            patt = "In this task, you will need to talk with our restaurants searching bot"
        elif slot_num == 1:
            slot_id = random.randint(0, 3)    
            ttask = random.sample(self.with_res, 1)[0]
            if slot_id == 0:
                price_range = ttask[0]; food_type = ""; area = ""
            elif slot_id == 1:
                price_range = ""; food_type = ttask[1]; area = ""
            else:
                price_range = ""; food_type = ""; area = "at the " + ttask[2] + " of the city"
            patt = task_patt.replace("<PRICE>", price_range).replace("<TYPE>", food_type).replace("<AREA>", area)
        elif slot_num == 2:
            slot_id = random.randint(0, 3)    
            ttask = random.sample(self.with_res, 1)[0]
            if slot_id == 0:
                price_range = ""; food_type = ttask[1]; area = "at the " + ttask[2] + " of the city"
            elif slot_id == 1:
                price_range = ttask[0]; food_type = ""; area = "at the " + ttask[2] + " of the city"
            else:
                price_range = ttask[0]; food_type = ttask[1]; area = ""
            patt = task_patt.replace("<PRICE>", price_range).replace("<TYPE>", food_type).replace("<AREA>", area)
        elif slot_num == 3:
            ttask = random.sample(self.with_res, 1)[0]
            price_range = ttask[0]; food_type = ttask[1]; area = "at the " + ttask[2] + " of the city"
            patt = task_patt.replace("<PRICE>", price_range).replace("<TYPE>", food_type).replace("<AREA>", area)
        else:
            ttask = random.sample(self.without_res, 1)[0]
            price_range = ttask[0]; food_type = ttask[1]; area = "at the " + ttask[2] + " of the city"
            patt = task_patt.replace("<PRICE>", price_range).replace("<TYPE>", food_type).replace("<AREA>", area)
            ttask = random.sample(self.with_res, 1)[0]
            price_range = ttask[0]; food_type = ttask[1]; area = "at the " + ttask[2] + " of the city"
            patt += task_2nd_patt.replace("<PRICE>", price_range).replace("<TYPE>", food_type).replace("<AREA>", area)
        inform_num = random.randint(1, 2)
        inform_l = random.sample(self.inform_list, inform_num)
        patt += ". You also want to know the " + ", ".join(inform_l) + " of this restaurant."
        patt += " If you want to quit the conversation please type in \"quit\". However, you still need to finish the follow-up questionnaire. When you finish the questionnaire, a blue button called \"Done with this HIT\" will show up. Please click it to finish the hit."
        return patt
        
    def task_generation(self):
        del self.keywords[:]
        task_patt = "In this task, you will need to talk with our restaurants searching bot. You want to find a restaurant that meets the following requirements:"
        task_2nd_patt = ". However, you may not find a restaurant on these conditions. You then change your mind to find a <PRICE> <TYPE> restaurant <AREA>"
        slot_num = random.randint(0, 5)
        price_range = ""; food_type = ""; area = ""
        if slot_num == 0:
            patt = "In this task, you will need to talk with our restaurants searching bot to search a restaurant. "
        if slot_num == 1:
            patt = "In this task, you will need to talk with our restaurants searching bot to search a restaurant based on location, food type and price range. "
        elif slot_num == 2:
            slot_id = random.randint(0, 3)    
            ttask = random.sample(self.with_res, 1)[0]
            reqs = []
            if slot_id == 0:
                price_range = ttask[0]
                reqs.append("Price range: " + price_range)
            elif slot_id == 1:
                food_type = ttask[1]
                reqs.append("Food type: " + food_type)
            else:
                area = ttask[2]
                reqs.append("Area: " + area)
            patt = task_patt + "\n<ul><li>" + "</li><li>".join(reqs) + "</li></ul>"
        elif slot_num == 3:
            slot_id = random.randint(0, 3)    
            ttask = random.sample(self.with_res, 1)[0]
            reqs = []
            if slot_id == 0:
                food_type = ttask[1]; area = ttask[2]
                reqs.append("Food type: " + food_type)
                reqs.append("Area: " + area)
            elif slot_id == 1:
                price_range = ttask[0]; area = ttask[2]
                reqs.append("Price range: " + price_range)
                reqs.append("Area: " + area)
            else:
                price_range = ttask[0]; food_type = ttask[1]
                reqs.append("Food type: " + food_type)
                reqs.append("Price range: " + price_range)
            random.shuffle(reqs)
            patt = task_patt + "\n<ul><li>" + "</li><li>".join(reqs) + "</li></ul>"
        elif slot_num == 4:
            ttask = random.sample(self.with_res, 1)[0]
            price_range = ttask[0]; food_type = ttask[1]; area = ttask[2]
            reqs = []
            reqs.append("Food type: " + food_type)
            reqs.append("Area: " + area)
            reqs.append("Price range: " + price_range)
            random.shuffle(reqs)
            
            patt = task_patt + "\n<ul><li>" + "</li><li>".join(reqs) + "</li></ul>"
        else:
            ttask = random.sample(self.without_res, 1)[0]
            price_range = ttask[0]; food_type = ttask[1]; area = ttask[2]
            reqs = []
            reqs.append("Food type: " + food_type)
            reqs.append("Area: " + area)
            reqs.append("Price range: " + price_range)
            random.shuffle(reqs)
            patt = task_patt + "\n<ul><li>" + "</li><li>".join(reqs) + "</li></ul>"
            ttask = random.sample(self.with_res, 1)[0]
            price_range = ttask[0]; food_type = ttask[1]; area = ttask[2]
            reqs = []
            reqs.append("Food type: " + food_type)
            reqs.append("Area: " + area)
            reqs.append("Price range: " + price_range)
            random.shuffle(reqs)
            patt += "\n" + task_2nd_patt + "\n<ul><li>" + "</li><li>".join(reqs) + "</li></ul>"
        inform_num = random.randint(1, 2)
        inform_l = random.sample(self.inform_list, inform_num)
        if price_range != "":
            self.keywords.append(price_range)
        if area != "":
            self.keywords.append(area)
        if food_type != "":
            self.keywords.append(food_type)
        self.keywords = self.keywords + inform_l
        patt += "You also want to know the " + ", ".join(inform_l) + " of this restaurant."
        patt += " If you want to quit the conversation please type in \"quit\". However, you still need to finish the follow-up questionnaire. When you finish the questionnaire, a blue button called \"Done with this HIT\" will show up. Please click it to finish the hit. \nAlso, during the conversation, please make a sentence as natural as possible. Please do not use only keywords."
        return patt
        

    def parley(self):
        # Each turn starts from the QA Collector agent
        del self.wrongturns[:]
        agent = DialogueAgent(hub_id='texthub')
        ad = {'episode_done': False}
        ad['id'] = self.__class__.collector_agent_id
        turn_num = 1

        ad['text'] = "<b>" + self.task_generation() + "</b>"
        self.mturk_agent.observe(validate(ad))

        sys_act = agent.start_call(session_id='texthub_dialog')
        ad['text'] = "<b>[Turn " + str(turn_num) + "] </b>" + sys_act.prompt
        self.mturk_agent.observe(validate(ad))

        while not agent.ENDING_DIALOG:
            turn_num += 1
            obs = self.mturk_agent.act()['text']
            loop_turns = 0
            while loop_turns < 3 and ((obs in self.keywords) or (len(obs.split(" ")) == 1 and obs != "quit")):
                ad['text'] = "Please make a sentence as natural as possible and do not use only keywords. Please rephrase your answer."
                self.mturk_agent.observe(validate(ad))
                obs = self.mturk_agent.act()['text']
                loop_turns += 1
            if loop_turns == 3:
                ad['text'] = "We are closing this HIT, since you were keeping texting in keywords and refused to rephrase your answer in a natural way. We are sorry that you can not be paid for this HIT."
                self.mturk_agent.observe(validate(ad))
                self.shutdown()
                self.mturk_agent.reject_work(reason="You were keeping texting in keywords and refused to rephrase your answer in a natural way.")
                return
            domain = None
            sys_act = agent.continue_call(asr_info = [(obs,1.0)], domainString = domain)
            ad['text'] = "<b>[Turn " + str(turn_num) + "] </b>" + sys_act.prompt
            if sys_act.prompt.find("having trouble understanding what you want") > -1:
                self.wrongturns.append(str(turn_num))
            self.mturk_agent.observe(validate(ad))

        ad['text'] = "Please tell me if you could find a restaurant that meets your constraints using our system and get all information (phone number/address) you need. (yes/no)"
        self.mturk_agent.observe(validate(ad))
        obs = self.mturk_agent.act()['text']

        ad['text'] = "Please tell me the turns you believe system's actions are contextually wrong(e.g. misunderstanding, questions repeating, information missing...), using turn numbers connected by \',\' (e.g. 1,3,6). If you think all turns are contextually right, please write 0. Please note that you can't get your reward without giving this feedback."

        loop_turns = 0
        while loop_turns < 3:
            loop_turns += 1
            self.mturk_agent.observe(validate(ad))
            obs = self.mturk_agent.act()['text']
            if self.pattern.match(obs) == None:
                ad['text'] = "Sorry the format of your feedback was wrong. Please tell me the turns you believe system's actions are contextually wrong(e.g. misunderstanding, questions repeating, information missing...), using turn numbers connected by \',\' (e.g. 1,3,6). If you think all turns are contextually right, please write 0. Please include those turns in your feedback if the system asked you for some information that you already provided."
                continue
            olist = [item.strip() for item in obs.split(",")]
            not_inclued = [item for item in self.wrongturns if item not in olist]
            if len(not_inclued) > 0:
                ad['text'] = "We detected that contextually wrong turns [" + ", ".join(not_inclued) + "] are not included in your feedback. Please resubmit your feedback."
                continue
            exceed_turns = [item for item in olist if int(item) > turn_num or int(item) < 0] 
            if len(exceed_turns) > 0:
                ad['text'] = "We detected that turns [" + ", ".join(exceed_turns) + "] in your feedback are illegal values. Please resubmit your feedback."
                continue
            break

        if loop_turns == 3:
            ad['text'] = "We are closing this HIT, since the format of your feedbacks were wrong and you refused to rephrase your feedback in a right way. We are sorry that you can not be paid for this HIT."
            self.mturk_agent.observe(validate(ad))
            self.shutdown()
            self.mturk_agent.reject_work(reason="The format of your feedbacks were wrong and you refused to rephrase your feedback in a right way.")
            return
        return False, ""

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        #self.task.shutdown()
        self.mturk_agent.shutdown()

    def review_work(self):
        pass
