import os
import sys

print('CWD:', os.getcwd())
print('sys.path[0]:', sys.path[0])
print('Exists src:', os.path.isdir('src'))
print('src contents:', os.listdir('src') if os.path.isdir('src') else None)

try:
    import src
    print('Imported src package:', src)
    import importlib
    print('src modules:', importlib.util.find_spec('src.preprocessing'))
except Exception as e:
    print('Import src failed:', e)
