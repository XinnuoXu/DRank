# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

task_config = {}


"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Talk with our restaurants booking bot and label the bad turns.'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'Talk with our restaurants booking bot and label the bad turns.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialogue system, task-based chatbot'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
'''
In this task, you will need to talk with our restaurants searching bot and tell us in which turns you think our bot's answer are inappropriate. <br><br>

Example:<br><br>
------------------- Task Begin ------------------- <br><br>
<b>Restaurant bot: In this task, you will need to talk with our restaurants searching bot. You want to find a cheap restaurant. </b><br><br>

<b>Restaurant bot</b>:[Turn 1] Hello, welcome to the restaurant finding system. How may I help you?<br><br>
<b>Worker</b>: Hi, I want to find a restaurant<br><br>

<b>Restaurant bot</b>:[Turn 2] What part of town do you have in mind?<br><br>
<b>Worker</b>: I'm looking for a restaurant in the city centre<br><br>

<b>Restaurant bot</b>:[Turn 3] What kind of food would you like?<br><br>
<b>Worker</b>: I prefer Chinese food<br><br>

<b>Restaurant bot</b>:[Turn 4] Would you like something in the cheap, moderate, or expensive price range?<br><br>
<b>Worker</b>: as cheap as possible<br><br>

<b>Restaurant bot</b>:[Turn 5] golden house is a nice place, it is in the cheap price range , it is in the centre of town and it serves chinese food.<br><br>
<b>Worker</b>: okay, thank you. bye<br><br>

<b>Restaurant bot</b>:[Turn 6]Thank you, goodbye.<br><br>

<b>Restaurant bot</b>: Please tell me if you could find a restaurant that meets your constraints using our system and get all information (phone number/address) you need.(yes/no)<br><br>
<b>Worker</b>: yes<br><br>

<b>Restaurant bot</b>: Please tell me the turns you believe system's actions are contextually wrong, using turn numbers connected by ',' (e.g. 1,3,6). If you are completely satisfied by this conversation, please write 0.<br><br>
<b>Worker</b>: 0<br><br>

If you want to quit the conversation please type in \"quit\". However, you still need to finish the follow-up questionnaire. When you finish the questionnaire, a blue button called \"Done with this HIT\" will show up. Please click it to finish the hit. <br><br>

------------------- Task Done ------------------- <br><br>
If you are ready, please click "Accept HIT" to start this task.
'''


