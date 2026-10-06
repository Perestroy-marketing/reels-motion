#!/usr/bin/env python3
"""Проверка окружения reels-motion: Python-модули, ffmpeg с нужными фильтрами и кодерами, Chromium для Playwright,
шрифты скилла (загрузка в Chromium с кириллицей), ключ fal для распознавания (предупреждение).

  check_env.py

На каждую проблему печатает команду установки. Код выхода 1, если что-то не так (FAL_KEY на код не влияет).
"""
import importlib
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = os.path.join(ROOT, 'requirements.txt')
MODULES = [('playwright', 'playwright'), ('numpy', 'numpy'), ('PIL', 'pillow'), ('scipy', 'scipy'), ('soundfile', 'soundfile'),
           ('cv2', 'opencv-python-headless'), ('psutil', 'psutil')]
FONTS = ['Inter-Black.otf', 'Inter-ExtraBold.otf', 'Inter-Bold.otf', 'Unbounded-900.ttf']
problems, warnings = [], []


def ok(m):
    print('  ок   ' + m)


def bad(m, fix):
    print('  НЕТ  ' + m)
    problems.append((m, fix))


def warn(m, fix):
    print('  ВНИМ ' + m)
    warnings.append((m, fix))


def ffmpeg_hint():
    s = platform.system()
    if s == 'Darwin':
        return 'brew install ffmpeg'
    if s == 'Windows':
        return 'winget install Gyan.FFmpeg'
    return 'sudo apt install ffmpeg   (Fedora: sudo dnf install ffmpeg, Arch: sudo pacman -S ffmpeg)'


def check_python():
    print('Python %s' % platform.python_version())
    if sys.version_info < (3, 10):
        bad('Python 3.10 или новее', 'поставьте Python 3.10+')
    for mod, pkg in MODULES:
        try:
            m = importlib.import_module(mod)
            ok('%s %s' % (pkg, getattr(m, '__version__', '')))
        except Exception as e:
            bad('модуль %s (%s)' % (pkg, e.__class__.__name__), 'python3 -m pip install -r "%s"' % REQ)


def check_ffmpeg():
    print('ffmpeg')
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            bad(tool + ' в PATH', ffmpeg_hint())
            return
    ok(subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.splitlines()[0])
    flt = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True).stdout
    enc = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    nf = {ln.split()[1] for ln in flt.splitlines() if len(ln.split()) > 2}
    ne = {ln.split()[1] for ln in enc.splitlines() if len(ln.split()) > 2}
    for f in ('alimiter', 'ebur128', 'loudnorm', 'sidechaincompress', 'acompressor', 'highpass', 'concat', 'trim', 'atrim', 'scale'):
        (ok if f in nf else lambda m: bad(m, 'нужен полный ffmpeg: ' + ffmpeg_hint()))('фильтр ' + f)
    for e in ('libx264', 'aac', 'qtrle'):
        (ok if e in ne else lambda m: bad(m, 'нужен ffmpeg с libx264 (сборка из репозитория ОС): ' + ffmpeg_hint()))('кодер ' + e)


FONT_JS = '''async (fams) => {
  const out = {};
  for (const f of fams) { const got = await document.fonts.load(f, 'АЯаяQ09№'); out[f] = got.length > 0 && document.fonts.check(f, 'Аа'); }
  return out;
}'''


def check_chromium_fonts():
    print('Chromium (Playwright) и шрифты')
    for f in FONTS:
        p = os.path.join(ROOT, 'fonts', f)
        (ok if os.path.exists(p) else lambda m: bad(m, 'файл шрифта пропал — переустановите скилл'))('шрифт ' + f)
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        bad('Playwright не импортируется — Chromium не проверить', 'python3 -m pip install playwright')
        return
    install = 'python3 -m playwright install chromium' + (
        '   (не хватает системных библиотек: sudo python3 -m playwright install-deps chromium)' if platform.system() == 'Linux' else '')
    css = ''.join("@font-face{font-family:'%s';font-weight:%s;src:url('file://%s');}" % (fam, w, os.path.join(ROOT, 'fonts', fn))
                  for fam, w, fn in (('RMInter', 900, 'Inter-Black.otf'), ('RMInter', 800, 'Inter-ExtraBold.otf'), ('RMInter', 700, 'Inter-Bold.otf'),
                                     ('RMUnbounded', 900, 'Unbounded-900.ttf')))
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(args=['--font-render-hinting=none', '--disable-lcd-text', '--allow-file-access-from-files'])
            try:
                pg = b.new_page(viewport={'width': 320, 'height': 240})
                import tempfile
                with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as fh:
                    fh.write('<!doctype html><meta charset=utf-8><style>%s</style><div>Аа</div>' % css)
                pg.goto('file://' + fh.name)
                os.unlink(fh.name)
                ok('Chromium %s запускается' % b.version)
                # свои имена семейств — чтобы не засчитать системный шрифт вместо файла скилла
                res = pg.evaluate(FONT_JS, ['900 40px RMInter', '800 40px RMInter', '700 40px RMInter', '900 40px RMUnbounded'])
                for f, good in res.items():
                    (ok if good else lambda m: bad(m, 'шрифт не загрузился в Chromium — проверьте папку fonts/'))('Chromium: ' + f + ' (кириллица)')
            finally:
                b.close()
    except Exception as e:
        bad('Chromium не запускается: %s' % str(e).strip().splitlines()[0], install)


def check_keys():
    print('Распознавание речи')
    try:
        import faster_whisper  # noqa: F401
        local = True
    except ImportError:
        local = False
    if os.environ.get('FAL_KEY'):
        ok('FAL_KEY задан (stt и qa --stt будут работать через fal)')
    elif local:
        ok('нет FAL_KEY — stt и qa --stt пойдут через локальный faster-whisper (бесплатно, офлайн)')
    else:
        warn('нет FAL_KEY и нет faster-whisper — шаги stt и qa --stt недоступны; остальной конвейер работает',
             'python3 -m pip install faster-whisper, или ключ fal (https://fal.ai) в FAL_KEY, или слова готовым JSON')


def main():
    check_python()
    check_ffmpeg()
    check_chromium_fonts()
    check_keys()
    print()
    if warnings:
        print('Предупреждений: %d (на код выхода не влияют):' % len(warnings))
        for m, fix in warnings:
            print('  - %s:\n      %s' % (m, fix))
    if problems:
        print('Проблем: %d. Что сделать:' % len(problems))
        for m, fix in problems:
            print('  - %s:\n      %s' % (m, fix))
        sys.exit(1)
    print('Окружение готово.')


if __name__ == '__main__':
    main()
