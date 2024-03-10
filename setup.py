from setuptools import setup, find_packages


setup(
    name='diffscore',
    version="0.0.1",
    packages=[package for package in find_packages()
              if package.startswith('diffscore')],
    install_requires=[
        'pytorch_lightning',
        # 'gym==0.25',  # TODO: conflict with neurogym's gym version compatibility
        'neurogym',
        'similarity-repository @ git+https://github.com/nacloos/similarity-repository.git'
    ],
    description='',
    author=''
)
