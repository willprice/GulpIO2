#!/usr/bin/env python
""" register an adapter to create a command line interface

Usage:
    register_adapter.py <adapter_file> <class_name>

Arguments:
    adapter_file:       Name of the adapter to register
    class_name:         Name of the new adapter class

"""
import os
import stat
import inspect
import importlib
from docopt import docopt

arguments = docopt(__doc__)
docopt_string = ''


def make_file_name(class_name):
    result = ''
    for letter in class_name:
        if letter.isupper():
            result += ("_" + letter.lower())
        else:
            result += letter
    return result


filename = make_file_name(arguments['<class_name>'])
if filename[0] == '_':
    filename = filename[1:]

i = importlib.import_module(arguments['<adapter_file>'])
init_args = inspect.getargspec(getattr(i, arguments['<class_name>']).__init__)
print(init_args)
parameters = init_args[0][1:]
defaults = ['' for i in range(len(parameters))]
defaults[-len(init_args[-1]):] = init_args[-1]

docopt_string += '#!/usr/bin/env python\n\n'
docopt_string += '"""' + filename
docopt_string += '\n\n'
docopt_string += 'Usage:\n'
docopt_string += ' ' * 4 + filename + '\n'

if 'videos_per_chunk' not in parameters:
    docopt_string += ' ' * (len(filename) + 8) + '[--videos_per_chunk <videos_per_chunk>]\n'
if 'num_workers' not in parameters:
    docopt_string += ' ' * (len(filename) + 8) + '[--num_workers <num_workers>]\n'

for parameter, default in zip(reversed(parameters), reversed(defaults)):
    if default == '':
        docopt_string += ' ' * (len(filename) + 8) + '<' + parameter + '>\n'
    else:
        docopt_string += ' ' * (len(filename) + 8) + '[--' + parameter + ' <' + parameter + '>]\n'

if 'output_folder' not in parameters:
    docopt_string += ' ' * (len(filename) + 8) + '<output_folder>\n'

docopt_string += '\n'

docopt_string += 'Options:\n'
for parameter, default in zip(parameters, defaults):
    if default != '':
        docopt_string += ' ' * 4 + '--' + parameter + '=<' + parameter + '>    [default: ' + str(default) + ']\n'

if 'videos_per_chunk' not in parameters:
    docopt_string += ' ' * 4 + '--videos_per_chunk=<videos_per_chunk>    [default: 100]\n'
if 'num_workers' not in parameters:
    docopt_string += ' ' * 4 + '--num_workers=<num_workers>    [default: 4]\n'
docopt_string += '"""\n\n'

docopt_string += 'from docopt import docopt\n'
docopt_string += 'from {} import {}\n'.format(arguments['<adapter_file>'],
                                              arguments['<class_name>'])
docopt_string += 'from gulpio.fileio import GulpIngestor\n\n'

docopt_string += 'if __name__ == "__main__":\n'
docopt_string += '    arguments = docopt(__doc__)\n'
docopt_string += '    print(arguments)\n\n'

for parameter, default in zip(parameters, defaults):
    if default == '':
        docopt_string += "    {} = arguments['<{}>']\n".format(parameter,
                                                               parameter)
    else:
        docopt_string += "    {} = arguments['--{}']\n".format(parameter,
                                                               parameter)


if 'output_folder' not in parameters:
    docopt_string += "    output_folder = arguments['<output_folder>']\n"
if 'videos_per_chunk' not in parameters:
    docopt_string += "    videos_per_chunk = arguments['--videos_per_chunk']\n"
if 'num_workers' not in parameters:
    docopt_string += "    num_workers = arguments['--num_workers']\n"

docopt_string += '\n'

docopt_string += "    adapter = " + arguments['<class_name>'] + '(\n'
for parameter in parameters:
    docopt_string += '        {},\n'.format(parameter)
docopt_string += '        )\n'

docopt_string += '    ingestor = GulpIngestor(adapter,\n' +\
                 '                            output_folder,\n' +\
                 '                            videos_per_chunk,\n' +\
                 '                            num_workers)\n' +\
                 '    ingestor()'


with open(filename, 'w') as fp:
    fp.write(docopt_string)
st = os.stat(filename)
os.chmod(filename, st.st_mode | stat.S_IEXEC)
print(docopt_string)
