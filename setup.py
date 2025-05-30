from setuptools import setup,find_packages
from typing import List

HYPHEN_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    ''' 
        This function will read the requirements from the file 
    '''
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('/n','') for req in requirements]
    
    if HYPHEN_DOT in requirements:
        requirements.remove(HYPHEN_DOT)


setup(
name='mlprojet',
version='0.0.1',
author='Hardik',
author_email='hardikkhobragade78@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)