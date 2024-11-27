if (not __import__('sys').warnoptions): __import__('warnings').filterwarnings('ignore', category=RuntimeWarning, module='runpy')
from .PySlang import *
del PySlang
