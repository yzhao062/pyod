# content of conftest.py
import sys
import platform

collect_ignore = []
# if sys.version_info[0] == 2 and 'linux' in platform.system():
if sys.version_info[0] == 2:
    # collect_ignore.append("test_so_gaal.py")
    # collect_ignore.append("test_mo_gaal.py")
    pass
