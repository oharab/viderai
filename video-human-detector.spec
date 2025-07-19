# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/video_human_detector/main.py'],
    pathex=[],
    binaries=[],
    datas=[('yolov8n.pt', '.')],
    hiddenimports=[
        'ultralytics',
        'cv2',
        'numpy',
        'click',
        'tqdm',
        'torch',
        'torchvision',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.models.yolo.predict',
        'ultralytics.engine.predictor',
        'ultralytics.utils',
        'ultralytics.data',
    ],
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
    name='video-human-detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
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
    name='video-human-detector',
)
