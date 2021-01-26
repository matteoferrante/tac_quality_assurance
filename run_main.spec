# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import sys
sys.setrecursionlimit(4000)

a = Analysis(['run_main.py'],
             pathex=['C:\\Users\\matte\\PycharmProjects\\streamlit_test'],
             binaries=[],
             datas=[(r"C:\Users\matte\miniconda3\envs\ieo_env\Lib\site-packages\altair\vegalite\v4\schema\vega-lite-schema.json",r"./altair/vegalite/v4/schema/"),(r"C:\Users\matte\miniconda3\envs\ieo_env\Lib\site-packages\streamlit\static",r"./streamlit/static")],
             hiddenimports=[],
             hookspath=['./hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='run_main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
