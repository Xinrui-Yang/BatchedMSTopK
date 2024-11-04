import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, _find_cuda_home

os.system('pip uninstall -y tcmm')

CUDA_DIR = _find_cuda_home()

# Python interface
setup(
    name='tcmm',
    version='0.2.0',
    install_requires=['torch'],
    packages=['tcmm'],
    package_dir={'tcmm': './'},
    ext_modules=[
        CUDAExtension(
            name='tcmm',
            include_dirs=['./', 
                CUDA_DIR+'/samples/common/inc'],
            sources=[
                'src/tcmm.cpp',
                'src/topk.cu',
                # 'src/batched_topk.cu',
            ],
            libraries=[],
            library_dirs=['objs', CUDA_DIR+'/lib64'],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Shaohuai Shi',
    author_email='shaohuais@cse.ust.hk',
    description='Efficient PyTorch Extension for TopK',
    keywords='Pytorch C++ Extension',
    zip_safe=False,
)