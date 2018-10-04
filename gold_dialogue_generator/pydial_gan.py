# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#import Texthub
import argparse, re, os, sys, random
from Agent import DialogueAgent, AgentFactory
from utils import ContextLogger
from utils import Settings
from ontology import Ontology

__author__ = "cued_dialogue_systems_group"
__version__ = Settings.__version__

global seed
seed = Settings.init(os.getcwd() + "/config/Tut-hdc-CamInfo.cfg", None)
global Ontology
Ontology.init_global_ontology()

class DialogueGenerator():
    
    #collector_agent_id = 'Restaurant bot'

    def __init__(self):
        self.agent = DialogueAgent(hub_id='texthub')

    def first_system_act(self):
        sys_act = self.agent.start_call(session_id='texthub_dialog')
        return sys_act.prompt

    def next_system_act(self, usr_utt, turn_num=1):
        if not self.agent.ENDING_DIALOG:
            if turn_num == 0:
                sys_act = self.agent.continue_call(asr_info = [(usr_utt,1.0)], domainString = None)
            else:
                sys_act = self.agent.continue_call(asr_info = [(usr_utt,1.0)], domainString = "CamRestaurants")
            return sys_act.prompt
        return ""

    def end_call(self):
        self.agent.end_call()

class DialogueManager():
    
    collector_agent_id = 'Restaurant bot'

    def __init__(self):
        self.agent = AgentFactory()

    def first_system_act(self, session_id):
        sys_act, agent_id = self.agent.start_call(session_id='texthub_dialog_' + session_id)
        return sys_act.prompt, agent_id

    def next_system_act(self, usr_utt, agent_id):
        if not self.agent.query_ENDING_DIALOG(agent_id):
            sys_act = self.agent.continue_call(agent_id, asr_info = [(usr_utt,1.0)], domainString = None)
            return sys_act.prompt
        return ""

if __name__ == '__main__':
    dg = DialogueGenerator()
    print (dg.first_system_act())
    print (dg.next_system_act("i would like an expensive indian restaurant in the east part of town"))
    print (dg.next_system_act("great , what ' s the address"))
    
    '''
    print (dg.next_system_act("hi , i want to book a restaurant", 0))
    print (dg.next_system_act("i prefer a venue in centre ."))
    #print (dg.next_system_act("i want a venue in the expensive price range serving british food ."))
    print (dg.next_system_act("i want a cheap restaurant serving chinese food ."))
    print (dg.next_system_act("What is the address?"))
    '''
