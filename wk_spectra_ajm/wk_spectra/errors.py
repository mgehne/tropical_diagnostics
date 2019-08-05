# -*- coding: utf-8 -*-

class Error(Exception):
    code = 100
    description = 'Base class for exceptions in this module'


class InputError(Error):
    code = 101
    description = "Exception raised for errors in the input"
