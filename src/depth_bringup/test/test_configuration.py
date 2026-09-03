import ast
from pathlib import Path
import unittest

import yaml


PACKAGE = Path(__file__).parents[1]


class ConfigurationTest(unittest.TestCase):
    def test_launch_files_have_valid_python_syntax(self):
        for path in (PACKAGE / 'launch').glob('*.launch.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_parameter_yaml_is_valid(self):
        for path in (PACKAGE / 'config').glob('*.yaml'):
            self.assertIsNotNone(yaml.safe_load(path.read_text()))

    def test_modes_and_safe_database_default_are_exposed(self):
        mapping = (PACKAGE / 'launch' / 'visual_slam_mapping.launch.py').read_text()
        full = (PACKAGE / 'launch' / 'visual_slam_full.launch.py').read_text()
        self.assertIn('UnlessCondition(use_mcu)', mapping)
        self.assertIn('IfCondition(use_mcu)', mapping)
        self.assertIn("'delete_db': (", mapping)
        self.assertIn("'false', 'Pass --delete_db_on_start", mapping)
        self.assertIn("'publish_mount_tf': 'true'", full)
