#!/usr/bin/env python

from setuptools import setup
from setuptools.command.install import install as _install

class install(_install):
    def pre_install_script(self):
        pass

    def post_install_script(self):
        pass

    def run(self):
        self.pre_install_script()

        _install.run(self)

        self.post_install_script()

if __name__ == '__main__':
    setup(
        name = 'gulpio',
        version = '155.0',
        description = '''Binary storage format for deep learning on videos.''',
        long_description = '''''',
        author = "Eren Golge, Raghav Goyal, Susanne Westphal, Heuna Kim, Guillaume Berger, Valentin Haenel",
        author_email = "eren.golge@twentybn.com, raghav.goyal@twentybn.com, susanne.westphal@twentybn.com, heuna.kim@twentybn.com, guillaume.berger@twentybn.com, valentin.haenel@twentybn.com",
        license = '',
        url = '',
        scripts = [
            'scripts/gulp_20bn_json_videos',
            'scripts/gulp_public_20bn_videos'
        ],
        packages = ['gulpio'],
        py_modules = [],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python'
        ],
        entry_points = {},
        data_files = [],
        package_data = {},
        install_requires = [
            'Pillow',
            'docopt',
            'opencv-python',
            'sh',
            'tqdm'
        ],
        dependency_links = [],
        zip_safe=True,
        cmdclass={'install': install},
    )
