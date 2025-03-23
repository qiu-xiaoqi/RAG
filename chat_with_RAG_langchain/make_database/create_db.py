import os
import sys
import re
# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# 打印当前路径
# print(os.getcwd())
import tempfile
from dotenv import load_dotenv, find_dotenv
from embedding.call_embedding import get_embedding