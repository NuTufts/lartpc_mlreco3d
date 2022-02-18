import os,sys

PARSE_MODULES = {}

def register_parser( name, parser_fn ):
    PARSE_MODULES[name] = parser_fn

def list_parsers():
    for name,parser_fn in PARSE_MODULES.items():
        print(name,": ",parser_fn)

def get_parser(name):
    if name not in PARSE_MODULES:
        raise ValueError(name," not registered")
    return PARSE_MODULES[name]
