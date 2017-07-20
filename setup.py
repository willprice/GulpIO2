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
        version = '1.0.dev0',
        description = '',
        long_description = '',
        author = '',
        author_email = '',
        license = '',
        url = '',
        scripts = [
            'scripts/create_binary_db_from_videos_skvideo.py',
            'scripts/create_binary_db_from_videos_ffmpeg_20bn_dataset.py',
            'scripts/create_binary_db.py',
            'scripts/create_binary_db_from_videos_ffmpeg.py',
            'scripts/.create_binary_db.py.swp',
            'scripts/create_binary_db_from_videos_ffmpeg_test_split.py',
            'scripts/make_parent_dict.py',
            'scripts/check_bin_file_sanity.py'
        ],
        packages = ['gulpio'],
        namespace_packages = [],
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
            'pandas',
            'sh',
            'tqdm'
        ],
        dependency_links = ['git+ssh://git@github.com/TwentyBN/20bn-research-tools.git'],
        zip_safe = True,
        cmdclass = {'install': install},
        keywords = '',
        python_requires = '',
        obsoletes = [],
    )
