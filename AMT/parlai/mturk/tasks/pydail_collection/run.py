# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.core.worlds import validate
from parlai.mturk.tasks.pydail_collection.worlds import \
    QADataCollectionOnboardWorld, QADataCollectionWorld
from parlai.mturk.core.mturk_manager import MTurkManager
from task_config import task_config
from utils import Settings, ContextLogger
import time
import os
import sys
import random
import importlib

logger = ContextLogger.getLogger('')

def main_bak():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    # Initialize a SQuAD teacher agent, which we will get context from
    module_name = 'parlai.tasks.squad.agents'
    class_name = 'DefaultTeacher'
    my_module = importlib.import_module(module_name)
    task_class = getattr(my_module, class_name)
    task_opt = opt.copy()
    task_opt['datatype'] = 'train'
    task_opt['datapath'] = opt['datapath']

    mturk_agent_id = 'Worker'
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=[mturk_agent_id])
    mturk_manager.setup_server()

    mturk_manager.set_onboard_function(onboard_function=None)
    mturk_manager.start_new_run()
    mturk_manager.create_hits()
    mturk_manager.ready_to_accept_workers()

    def check_workers_eligibility(workers):
        return workers

    eligibility_function = {'func': check_workers_eligibility, 'multiple': True,}

    def assign_worker_roles(worker):
        worker[0].id = mturk_agent_id

    global run_conversation
    def run_conversation(mturk_manager, opt, workers):
        mturk_agent = workers[0]
        world = QADataCollectionWorld(opt=opt, mturk_agent=mturk_agent)
        btime = time.time()
        world.parley()
        etime = time.time()
        logger.debug("DialTime: ", (etime - btime))
        world.shutdown()
        world.review_work()

    mturk_manager.start_task(
        eligibility_function=eligibility_function,
        assign_role_function=assign_worker_roles,
        task_function=run_conversation
    )

    mturk_manager.expire_all_unassigned_hits()
    mturk_manager.shutdown()


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)
    user_ids = {}

    # Initialize a SQuAD teacher agent, which we will get context from
    module_name = 'parlai.tasks.squad.agents'
    class_name = 'DefaultTeacher'
    my_module = importlib.import_module(module_name)
    task_class = getattr(my_module, class_name)
    task_opt = opt.copy()
    task_opt['datatype'] = 'train'
    task_opt['datapath'] = opt['datapath']

    mturk_agent_id = 'Worker'
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=[mturk_agent_id])
    mturk_manager.setup_server()

    mturk_manager.set_onboard_function(onboard_function=None)
    mturk_manager.start_new_run()
    mturk_manager.create_hits()
    mturk_manager.ready_to_accept_workers()

    def check_workers_eligibility(workers):
        return workers

    eligibility_function = {'func': check_workers_eligibility, 'multiple': True,}

    def assign_worker_roles(worker):
        worker[0].id = mturk_agent_id

    global run_conversation
    def run_conversation(mturk_manager, opt, workers):
        mturk_manager.left_pane_refresh(task_config['task_description'])
        mturk_agent = workers[0]
        if (mturk_agent.worker_id in user_ids):
            print ("USER_ID: ", mturk_agent.worker_id, " DIALS: ", user_ids[mturk_agent.worker_id])
        else:
            print ("USER_ID: ", mturk_agent.worker_id, " DIALS: 0")
        if (mturk_agent.worker_id in user_ids) and user_ids[mturk_agent.worker_id] >= 15:
            ad = {'episode_done': False}
            ad['id'] = 'Restaurant bot'
            ad['text'] = "We are closing this HIT, since you've already had over 15 dialogues with our restaurant bot in this session. We are very appreciated for your help. Welcome to join the next session."
            mturk_agent.observe(validate(ad))
            return
        else:
            world = QADataCollectionWorld(opt=opt, mturk_agent=mturk_agent)
            btime = time.time()
            world.parley()
            etime = time.time()
            logger.debug("DialTime: " + str(etime - btime))
            if mturk_agent.worker_id not in user_ids:
                user_ids[mturk_agent.worker_id] = 1
            else:
                user_ids[mturk_agent.worker_id] += 1
            world.shutdown()
            world.review_work()
            return 


    mturk_manager.start_task(
        eligibility_function=eligibility_function,
        assign_role_function=assign_worker_roles,
        task_function=run_conversation
    )

    mturk_manager.expire_all_unassigned_hits()
    mturk_manager.shutdown()

if __name__ == '__main__':
    main()
