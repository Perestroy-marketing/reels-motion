"""Распознавание речи по словам: fal whisper (fal-ai/whisper, ключ в FAL_KEY) или локальный faster-whisper.

Движок выбирает переменная RM_STT: `fal` или `local`. Если она не задана — fal при наличии FAL_KEY, иначе локально.
Локальный движок бесплатный и офлайн (модель — RM_WHISPER_MODEL, по умолчанию small; первый запуск её скачает).
Оба движка отдают один формат: {"text": "...", "chunks": [{"timestamp": [s, e], "text": " слово"}]} —
его же можно положить готовым JSON, тогда распознавание не нужно вовсе.
"""
import base64
import json
import os
import subprocess
import urllib.error
import urllib.request

from .common import die, log

ENDPOINT = 'https://fal.run/fal-ai/whisper'
LONG_WORD = 1.2      # слово длиннее — внутри почти всегда спрятан повтор или оговорка


def fal_key():
    k = os.environ.get('FAL_KEY', '').strip()
    if not k:
        die('нет переменной окружения FAL_KEY — распознавание пропущено. Задайте ключ fal в переменной окружения FAL_KEY '
            'или положите слова готовым JSON в формате fal chunks.', 2)
    return k


def audio_mp3(path, start=None, end=None, tempo=None):
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None:
        cmd += ['-ss', '%.3f' % start]
    cmd += ['-i', path]
    if end is not None:
        cmd += ['-t', '%.3f' % (end - (start or 0))]
    af = ['aresample=24000']
    if tempo:
        af.append('atempo=%g' % tempo)
    cmd += ['-vn', '-ac', '1', '-af', ','.join(af), '-b:a', '64k', '-f', 'mp3', '-']
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def engine():
    e = os.environ.get('RM_STT', '').strip().lower()
    if e in ('fal', 'local'):
        return e
    return 'fal' if os.environ.get('FAL_KEY', '').strip() else 'local'


_MODEL = None


def transcribe_local(mp3, level='word', lang='ru'):
    """faster-whisper на CPU; результат в формате fal chunks."""
    global _MODEL
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        die('нет FAL_KEY и не установлен faster-whisper: python3 -m pip install faster-whisper '
            '(или задайте FAL_KEY, или положите слова готовым JSON в формате fal chunks)', 2)
    if _MODEL is None:
        name = os.environ.get('RM_WHISPER_MODEL', 'small')
        log('локальное распознавание: faster-whisper %s (первый запуск скачает модель)' % name)
        _MODEL = WhisperModel(name, device='cpu', compute_type='int8')
    # декодируем сами через ffmpeg: так не зависим от версии PyAV внутри faster-whisper
    import numpy as np
    pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
                         input=mp3, capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    segs, _ = _MODEL.transcribe(audio, language=lang, word_timestamps=(level == 'word'),
                                vad_filter=False, condition_on_previous_text=False)
    segs = list(segs)
    chunks = []
    for s in segs:
        if level == 'word':
            for w in s.words or []:
                chunks.append({'timestamp': [round(w.start, 3), round(w.end, 3)], 'text': ' ' + w.word.strip()})
        else:
            chunks.append({'timestamp': [round(s.start, 3), round(s.end, 3)], 'text': ' ' + s.text.strip()})
    return {'text': ''.join(c['text'] for c in chunks).strip(), 'chunks': chunks}


def transcribe(mp3, level='word', lang='ru', timeout=300):
    if engine() == 'local':
        return transcribe_local(mp3, level, lang)
    body = json.dumps({'audio_url': 'data:audio/mpeg;base64,' + base64.b64encode(mp3).decode(), 'task': 'transcribe',
                       'language': lang, 'chunk_level': level}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={'Authorization': 'Key ' + fal_key(), 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        die('fal whisper ответил %s: %s' % (e.code, e.read()[:300].decode('utf-8', 'replace')))


def report(name, j):
    ch = j.get('chunks', [])
    print('== %s: %d фрагм.' % (name, len(ch)))
    print((j.get('text') or '').strip())
    long = [(c['text'].strip(), c['timestamp'][0], c['timestamp'][1]) for c in ch
            if c['timestamp'][1] is not None and c['timestamp'][1] - c['timestamp'][0] > LONG_WORD]
    for w, s, e in long:
        print('  ⚠ длинное слово «%s» %.2f–%.2f (%.2f с) — проверить замедленной перерасшифровкой: --slow %.2f:%.2f'
              % (w, s, e, e - s, max(0, s - 0.3), e + 0.3))
    print()


def main(a):
    os.makedirs(a.out, exist_ok=True)
    if a.slow:
        if len(a.files) != 1:
            die('--slow — для одного файла')
        s, e = (float(x) for x in a.slow.split(':'))
        src = a.files[0]
        name = os.path.splitext(os.path.basename(src))[0]
        j = transcribe(audio_mp3(src, s, e, tempo=0.5), a.level, a.lang)
        for c in j.get('chunks', []):             # шкала замедленной ×2 → шкала исходника
            t0, t1 = c['timestamp']
            c['timestamp'] = [round(s + t0 / 2, 3), None if t1 is None else round(s + t1 / 2, 3)]
        out = os.path.join(a.out, '%s_slow_%.2f_%.2f.json' % (name, s, e))
        json.dump(j, open(out, 'w'), ensure_ascii=False, indent=1)
        report(name + ' (×2 замедленно, время исходника)', j)
        for c in j.get('chunks', []):
            print('  %7.2f–%7s  %s' % (c['timestamp'][0], '%.2f' % c['timestamp'][1] if c['timestamp'][1] is not None else '?', c['text'].strip()))
        log('→ ' + out)
        return
    for src in a.files:
        name = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(a.out, name + '.json')
        if os.path.exists(out) and not a.force:
            j = json.load(open(out, encoding='utf-8'))
        else:
            j = transcribe(audio_mp3(src), a.level, a.lang)
            json.dump(j, open(out, 'w'), ensure_ascii=False, indent=1)
        report(name, j)
        log('→ ' + out)
