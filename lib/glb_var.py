# @Time   : 2022/4/12
# @Author : Junwei Lei
# @Email  : darrius.lei@outlook.com

from lib.callback import CustomException
import pydash as ps
'''
Store global variables for the project

Methods:
--------

__init__():None
    initialization

set_value(key, value):None
    set global variables

get_value(key):value
    retrieve global variables

Examples:
---------
@main.py
>>> import lib.glb_var
>>> glb_var.set_value('a', 1);
>>> submain;

@submain.py
>>> import lib.glb_var
>>> print(glb_var.get_value('a'));

'''
def __init__():
    global glb_dict;
    glb_dict = {};

def set_values(dict, keys = None):
    if keys is not None:
        dict = ps.pick(dict, keys);
    for key, value in dict.items():
        glb_dict[key] = value;

def set_value(key, value):
    #set global var
    glb_dict[key] = value;

def get_value(key):
    try:
        return glb_dict[key];
    except KeyError:
        raise CustomException(f'The retrieved key [{key}] does not exist');