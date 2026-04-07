from setuptools import setup
import os
from glob import glob

package_name = 'viplanner_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RPL-CS-UCL',
    maintainer_email='jlcucumber@ucl.ac.uk',
    description='ROS2 wrapper for VIPlanner local path planner',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'viplanner_node = viplanner_ros2.viplanner_node:main',
        ],
    },
)
