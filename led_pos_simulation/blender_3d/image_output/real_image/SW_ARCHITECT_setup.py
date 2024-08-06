from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "SW_ARCHITECT_lib",                 # 확장 모듈 이름
        sources=["SW_ARCHITECT_LIBRARY.pyx"],   # Cython 소스 파일
        include_dirs=[numpy.get_include()],  # numpy 헤더 파일 경로 포함
        # libraries=["m"],            # 수학 라이브러리 링크 (필요한 경우)
    )
]

setup(
    name="SW_ARCHITECT_lib",
    ext_modules=cythonize(ext_modules),
)