from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='D_GLSNet',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='ext',
            sources=[
                'extensions/extra/cloud/cloud.cpp',
                'extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
