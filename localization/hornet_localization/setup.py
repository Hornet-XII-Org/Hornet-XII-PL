from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'hornet_localization'


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nix Koh',
    maintainer_email='nyxkk@users.noreply.github.com',
    description='Initial IMU localization integration for Hornet XII.',
    license='LicenseRef-Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_localization_node = '
            'hornet_localization.imu_localization_node:main',
            'yaw_aligner_node = '
            'hornet_localization.yaw_aligner_node:main',
            'sim_imu_adapter_node = '
            'hornet_localization.sim_imu_adapter_node:main',
        ],
    },
)
