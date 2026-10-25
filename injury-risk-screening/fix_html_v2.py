"""
Fix remaining garbled characters in index.html.
The file is valid UTF-8 but contains double-encoded sequences.
Emoji like U+1F3E5 (hospital) was encoded to UTF-8 bytes F0 9F 8F A5,
then those bytes were re-interpreted as Latin-1 and re-encoded as UTF-8,
yielding: C3 B0 C5 B8 C2 8F C2 A5 = "ðŸ\x8f¥"

Strategy: read file as bytes, detect and replace all double-encoded sequences.
"""
import re

REPLACEMENTS = [
    # Emojis that appear garbled — map double-encoded UTF-8 bytes to correct HTML entities
    # U+1F3E5 hospital emoji: F0 9F 8F A5 -> double-encoded as C3B0 C59F C28F C2A5
    (b'\xc3\xb0\xc5\xb8\xc2\x8f\xc2\xa5', b'&#x1F3E5;'),   # 🏥
    # U+1F4CA chart: F0 9F 93 8A -> C3B0 C59F C293 C28A
    (b'\xc3\xb0\xc5\xb8\xc2\x93\xc2\x8a', b'&#x1F4CA;'),   # 📊
    # U+1F50D magnifier: F0 9F 94 8D -> C3B0 C59F C294 C28D
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x8d', b'&#x1F50D;'),   # 🔍
    # U+1F4A7 droplet: F0 9F 92 A7 -> C3B0 C59F C292 C2A7
    (b'\xc3\xb0\xc5\xb8\xc2\x92\xc2\xa7', b'&#x1F4A7;'),   # 💧
    # U+1F489 syringe: F0 9F 92 89 -> C3B0 C59F C292 C289
    (b'\xc3\xb0\xc5\xb8\xc2\x92\xc2\x89', b'&#x1F489;'),   # 💉
    # U+1F955 carrot: F0 9F A5 95 -> C3B0 C59F C2A5 C295
    (b'\xc3\xb0\xc5\xb8\xc2\xa5\xc2\x95', b'&#x1F955;'),   # 🥕
    # U+1F3CB weightlifter: F0 9F 8F 8B -> C3B0 C59F C28F C28B
    (b'\xc3\xb0\xc5\xb8\xc2\x8f\xc2\x8b', b'&#x1F3CB;'),   # 🏋
    # U+1F4B9 chart up: F0 9F 92 B9 -> C3B0 C59F C292 C2B9
    (b'\xc3\xb0\xc5\xb8\xc2\x92\xc2\xb9', b'&#x1F4B9;'),   # 💹
    # U+2713 check mark: E2 9C 93 -> C3A2 C29C C293
    (b'\xc3\xa2\xc2\x9c\xc2\x93', b'&#x2713;'),             # ✓
    # U+2022 bullet: E2 80 A2 -> C3A2 C280 C2A2
    (b'\xc3\xa2\xc2\x80\xc2\xa2', b'&bull;'),               # •
    # U+2026 ellipsis: E2 80 A6 -> C3A2 C280 C2A6
    (b'\xc3\xa2\xc2\x80\xc2\xa6', b'&hellip;'),             # …
    # U+201C ldquo: E2 80 9C -> C3A2 C280 C29C
    (b'\xc3\xa2\xc2\x80\xc2\x9c', b'&ldquo;'),             # "
    # U+201D rdquo: E2 80 9D -> C3A2 C280 C29D
    (b'\xc3\xa2\xc2\x80\xc2\x9d', b'&rdquo;'),             # "
    # U+00B7 middle dot: C2 B7 (already correct UTF-8, no double-encoding)
    # U+00B0 degree: C2 B0 (already correct UTF-8)
    # U+FFFD replacement char: EF BF BD -> might appear as C3AF C2BF C2BD
    (b'\xc3\xaf\xc2\xbf\xc2\xbd', b''),                    # remove replacement chars
    # Stray byte sequences from partial emoji: C3 B0 C5 B8 alone (orphaned 4-byte emoji start)
]

# Additional targeted string replacements (UTF-8)
STRING_REPLACEMENTS = [
    # Fix the "Loading…" badge that shows garbled
    ('Loading\u00e2\u0082\u00a6', 'Loading&hellip;'),
    # Fix "teach • demo • track" if garbled
    ('â\x82¢', '&bull;'),
    # Fix remaining garbled emoji in recovery section
    ('\u00c3\u00b0\u00c5\u00b8\u00c2\x8f\u00c2\xa5', '&#x1F3E5;'),
]

with open('src/web/index.html', 'rb') as f:
    raw = f.read()

print(f'Original size: {len(raw)}')

# Apply byte-level replacements (longest first to avoid partial matches)
for old, new in sorted(REPLACEMENTS, key=lambda x: len(x[0]), reverse=True):
    count = raw.count(old)
    if count > 0:
        print(f'  Replacing {old.hex()} ({count}x) -> {new.decode()}')
        raw = raw.replace(old, new)

# Decode and check
try:
    text = raw.decode('utf-8')
    print(f'UTF-8 decode OK. Size: {len(text)}')
except UnicodeDecodeError as e:
    print(f'UTF-8 decode error after fix: {e}')
    # Fall back to latin-1
    text = raw.decode('latin-1')

# Apply any string-level replacements
for old, new in STRING_REPLACEMENTS:
    if old in text:
        count = text.count(old)
        print(f'  String replacing {repr(old[:20])} ({count}x)')
        text = text.replace(old, new)

# Write back
with open('src/web/index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done. Verifying...')
with open('src/web/index.html', 'rb') as f:
    verify = f.read()

# Check for remaining multi-byte garbled sequences
problematic = []
i = 0
while i < len(verify) - 1:
    # Double-encoded 4-byte emoji: starts with 0xC3 0xB0
    if verify[i] == 0xC3 and verify[i+1] == 0xB0 and i+3 < len(verify):
        problematic.append((i, verify[i:i+8].hex()))
    i += 1
print(f'Remaining double-encoded 4-byte sequences: {len(problematic)}')
if problematic[:5]:
    print('  First few:', problematic[:5])
