"""

Interface for python package installation

"""

import setuptools


with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

with open('requirements.txt', 'r') as _requirements:
    requires = _requirements.read()

requires = [r.strip() for r in requires.split('\n') if ((r.strip()[0] != "#") and (len(r.strip()) > 3) and "-e git://" not in r)]

setuptools.setup(
    name='kfp_v1_ml_ops',
    version='0.0.1',
    author='Gianni Balistreri',
    author_email='gbalistreri762@gmail.com',
    description='Toolbox for easy and effective ml-ops deployment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='machine-learning ml-ops artificial-intelligence data-science kubeflow kubernetes',
    license='GNU',
    url='https://gitlab.shopware.com/shopware/machine-learning/ml-ops/kubeflow-templates',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'kfp_v1_ml_ops': ['LICENSE',
                                    'README.md',
                                    'requirements.txt',
                                    'setup.py'
                                    ]
                  },
    #data_file=[('test', [])],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requires
)
