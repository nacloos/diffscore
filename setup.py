from setuptools import setup, find_packages


setup(
    name='diffscore',
    version="0.0.1",
    packages=[package for package in find_packages()
              if package.startswith('diffscore')],
    install_requires=[
        'pytorch_lightning',
        'gdown',
        'neurogym',
        # TODO: error
        # 'similarity-repository @ git+https://github.com/nacloos/similarity-repository.git'
    ],
    description='',
    author=''
)
