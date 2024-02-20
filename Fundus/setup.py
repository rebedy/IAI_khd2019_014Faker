#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2
from distutils.core import setup

print('setup_test.py is running...')

setup(name='KYD_2019_Fundus',
      version='1.0',
      install_requires=['scikit-learn']
      )
