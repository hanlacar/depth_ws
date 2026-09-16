from setuptools import setup
from glob import glob
import os

package_name = 't870_mcu_simple'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='URRC',
    maintainer_email='urrc@example.com',
    description='Minimal T870 MCU direct bridge',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bridge = t870_mcu_simple.bridge_node:main',
        ],
    },
)
