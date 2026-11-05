from glob import glob
from setuptools import find_packages, setup

setup(
    name='hornet_front_camera', version='0.1.0', packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hornet_front_camera']),
        ('share/hornet_front_camera', ['package.xml']),
        ('share/hornet_front_camera/launch', glob('launch/*.launch.py')),
        ('share/hornet_front_camera/config', glob('config/*.yaml')),
        ('share/hornet_front_camera/config/examples', glob('config/examples/*.yaml')),
        ('share/hornet_front_camera/config/measured', glob('config/measured/*.yaml') + glob('config/measured/*.md')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Hornet XII PL subteam', maintainer_email='maintainers@example.invalid',
    description='Offline calibration and runtime rectification', license='LicenseRef-Proprietary',
    entry_points={'console_scripts': [
        'front_calibration = hornet_front_camera.calibration:main',
        'calibrate_stereo = hornet_front_camera.calibrate:main',
    ]},
)
