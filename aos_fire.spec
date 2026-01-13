# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = []

tmp_datas, tmp_binaries, tmp_hidden = collect_all('depth_anything_3')
datas += tmp_datas
binaries += tmp_binaries
hiddenimports += tmp_hidden

for package in ['safetensors', 'huggingface_hub', 'imageio', 'moviepy', 'tqdm']:
    datas += copy_metadata(package)

hiddenimports += ['safetensors.torch', 'sklearn.utils._typedefs']

host_os_files = [] 

a = Analysis(
    ['LFR_executable.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + host_os_files,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='aos_fire',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='aos_fire',
)