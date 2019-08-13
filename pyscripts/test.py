"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from __future__ import print_function, unicode_literals

from pprint import pprint

from PyInquirer import style_from_dict, Token, prompt

questions = [
    {
        'type': 'checkbox',
        'message': 'Select toppings',
        'name': 'toppings',
        'choices': [ 
            {
                'name': 'Ham',
                'value': 1
            },
        ],
        'validate': lambda answer: 'You must choose at least one process.' \
            if len(answer) == 0 else True
    }
]

answers = prompt(questions)
pprint(1 in answers['toppings'])