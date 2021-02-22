from setuptools import setup, find_packages

setup(
  name = 'pi-gan-pytorch',
  packages = find_packages(),
  version = '0.0.11',
  license='MIT',
  description = 'π-GAN - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/pi-gan-pytorch',
  keywords = [
    'artificial intelligence',
    'generative adversarial network'
  ],
  install_requires=[
    'einops>=0.3',
    'pillow',
    'torch>=1.6',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
