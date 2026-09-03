from glob import glob
from setuptools import setup

package_name = 'depth_bringup'
setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='hanlacar',
    maintainer_email='hanlacar@users.noreply.github.com',
    description='D456 RTAB-Map launch package.',
    license='Apache-2.0',
)
