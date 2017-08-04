from pybuilder.core import use_plugin, init, Author
from pybuilder.vcs import count_travis

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.install_dependencies")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin("filter_resources")


name = "gulpio"
default_task = "publish"
version = count_travis()
summary = "Binary storage format for deep learning on videos."
authors = [Author("Eren Golge", "eren.golge@twentybn.com"),
           Author("Raghav Goyal", "raghav.goyal@twentybn.com"),
           Author("Susanne Westphal", "susanne.westphal@twentybn.com"),
           Author("Heuna Kim", "heuna.kim@twentybn.com"),
           Author("Guillaume Berger", "guillaume.berger@twentybn.com"),
           Author("Valentin Haenel", "valentin.haenel@twentybn.com"),
           ]

requires_python = ">=3.4"


@init
def set_properties(project):
    project.depends_on('tqdm')
    project.depends_on('opencv-python')
    project.depends_on('docopt')
    project.depends_on('Pillow')
    project.depends_on('sh')
    project.get_property('filter_resources_glob').extend(
        ['**/gulpio/__init__.py'])
    project.get_property('coverage_exceptions').extend(
        ['gulpio.adapters'])
