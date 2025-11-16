# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect lightning_fabric data (existing)
datas = collect_data_files('lightning_fabric')

datas += collect_data_files('easynmt')
easynmt_hiddenimports = collect_submodules('easynmt')

datas += collect_data_files('faster_whisper')
faster_whisper_hiddenimports = collect_submodules('faster_whisper')

a = Analysis(
    ['whisperworker.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=easynmt_hiddenimports + faster_whisper_hiddenimports,
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
    a.binaries,
    a.datas,
    [],
    name='whisperworker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['book.ico'],
)