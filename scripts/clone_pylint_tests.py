# -*- coding: utf-8 -*-
"""
Clone pylint tests from https://github.com/PyCQA/pylint

Script used in tox tests
"""

import os

from git import Repo

tox_test_data_dir = os.environ['TOX_TEST_DATA_DIR']
build_dir = os.environ['PACKAGE_ROOT']

if not os.path.isdir(os.path.join(tox_test_data_dir, ".git")):
    Repo.clone_from(
        'https://github.com/aperezhortal/pylint',
        tox_test_data_dir,
        branch='only_tests',
        depth=1)
else:
    test_data_repo = Repo(tox_test_data_dir)
    test_data_repo.remotes['origin'].pull()
