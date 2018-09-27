import textwrap
import re
import os
import subprocess
import sys

optionWrapper = textwrap.TextWrapper(initial_indent='    ',
                                     subsequent_indent='    ',
                                     width=72)


def listify(value):
    """
    Convert an option specified as a string to a list.  Allow both
    comma and space as delimiters. Passes lists transparently.
    """
    if isinstance(value, str):
        return value.replace(',', ' ').split()
    elif isinstance(value, list):
        out_list = []
        for val in value:
            if isinstance(val, list):
                out_list.extend(listify(val))
            else:
                out_list.append(val)
        return out_list


def formatOption(env, opt):
    """
    Print a nicely formatted description of a SCons configuration
    option, its permitted values, default value, and current value
    if different from the default.
    """
    # Extract the help description from the permitted values. Original format
    # is in the format: "Help text ( value1|value2 )" or "Help text"
    if opt.help.endswith(')'):
        parts = opt.help.split('(')
        help = '('.join(parts[:-1])
        values = parts[-1][:-1].strip().replace('|', ' | ')
        if values == '':
            values = 'string'
    else:
        help = opt.help
        values = 'string'

    # Fix the representation of boolean options, which are stored as
    # Python bools, but need to be passed by the user as strings
    default = opt.default
    if default is True:
        default = 'yes'
    elif default is False:
        default = 'no'

    # First line: "* option-name: [ choice1 | choice2 ]"
    lines = ['* %s: [ %s ]' % (opt.key, values)]

    # Help text, wrapped and idented 4 spaces
    lines.extend(optionWrapper.wrap(re.sub(r'\s+', ' ', help)))

    # Default value
    lines.append('    - default: %r' % default)

    # Get the actual value in the current environment
    if opt.key in env:
        actual = env.subst('${%s}' % opt.key)
    else:
        actual = None

    # Fix the representation of boolean options
    if actual == 'True':
        actual = 'yes'
    elif actual == 'False':
        actual = 'no'

    # Print the value if it differs from the default
    if actual != default:
        lines.append('    - actual: %r' % actual)
    lines.append('')

    return lines


def getCommandOutput(cmd, *args):
    """
    Run a command with arguments and return its output.
    """
    environ = dict(os.environ)
    if 'PYTHONHOME' in environ:
        # Can cause problems when trying to run a different Python interpreter
        del environ['PYTHONHOME']
    data = subprocess.check_output([cmd] + list(args), env=environ)
    if sys.version_info.major == 3:
        return data.strip().decode('utf-8')
    else:
        return data.strip()
