from setuptools import setup, Extension
import os

# 현재 디렉토리의 절대 경로를 얻습니다.
current_dir = os.path.abspath(os.path.dirname(__file__))

setup(
    name='lamda_pnp_solver',
    version='0.1',
    ext_modules=[
        Extension(
            'lamda_pnp_solver',
            ['pnp_python_binding.cpp'],
            include_dirs=[current_dir, '/usr/local/lib/python3.8/dist-packages/pybind11/include'],  # 헤더 파일 검색 경로를 추가합니다.
            library_dirs=[current_dir + '/build/'],
            libraries=['libpnp', 'ceres'],  # 'libpnp' 라이브러리를 링크합니다. 라이브러리의 위치가 표준 경로에 없다면 'library_dirs' 옵션도 사용해야 할 수 있습니다.
            language='c++'
        ),
    ],
)

