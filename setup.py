from setuptools import setup, find_packages


setup(
    name='diffscore',
    version="0.0.1",
    packages=[package for package in find_packages()
              if package.startswith('diffscore')],
    install_requires=[
        'pytorch_lightning',
        'neurogym==0.0.2',
        'numpy<2.0.0',
        # 'brainscore_vision @ git+https://github.com/brain-score/vision',
        'similarity-repository @ git+https://github.com/nacloos/similarity-repository.git'
    ],
    description='',
    author=''
)
