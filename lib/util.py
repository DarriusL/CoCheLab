import pydash as ps
import datetime, time, torch

def set_attr(obj, dict, keys = None):
    '''Quickly assign properties to objects

    Parameters:
    -----------
    obj: object of a python class
        Assign values to the properties of the object

    dict: dict
        A dictionary for assigning values to object attributes, 
        whose key is the attribute, and the corresponding value is the value
    
    keys: list, optional
        defalue: None
        Select the corresponding key in the dictionary to assign a value to the object

    '''
    if keys is not None:
        dict = ps.pick(dict, keys);
    for key, value in dict.items():
        setattr(obj, key, value);

def get_attr(obj, keys):
    '''Quick access to object properties

    Parameters:
    -----------

    obj: object of a python class
        Get the properties of the object
    
    keys: list
        A list of properties of the object that require
    
    Returns:
    --------

    dict: dict
        The required attribute is the value corresponding to the keys of the dictionary
    

    '''
    dict = dict();
    for key in keys:
        if hasattr(obj, key):
            dict[key] = getattr(obj, key);
    return dict;

def get_date(separator = '_'):
    '''Quick get formated date
    '''
    return str(datetime.date.today()).replace('-', separator);

def s2hms(s):
    '''Convert s to hms
    '''
    return time.strftime("%H hour - %M min - %S s", time.gmtime(s));
