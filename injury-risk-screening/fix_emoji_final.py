"""
Final fix for remaining garbled emoji in index.html.
Each problematic sequence is a triple-encoded 4-byte emoji.
We replace each exact byte pattern with the correct HTML entity.
"""

with open('src/web/index.html', 'rb') as f:
    raw = f.read()

print(f'File size: {len(raw)}')

# Map each garbled byte sequence to the correct HTML entity
# Context: 
#   c3b0c5b8e2809cc5a0 before "Inspect 22 Biomechanical" -> chart U+1F4CA 📊
#   c3b0c5b8e28099c2a7 before "Daily Hydration" -> droplet U+1F4A7 💧
#   c3b0c5b8e28099c5a0 before "Active Medications" -> syringe U+1F489 💉
#   c3b0c5b8c2a5e28094 before "Active Nutrition" -> carrot/salad U+1F955 🥕 or U+1F96C 🥬
#   c3b0c5b8c28fc692 before "Physical Therapy" -> weightlifter U+1F3CB 🏋

BYTE_FIXES = [
    (bytes.fromhex('c3b0c5b8e2809cc5a0'), b'&#x1F4CA;'),   # 📊 chart
    (bytes.fromhex('c3b0c5b8e28099c2a7'), b'&#x1F4A7;'),   # 💧 droplet
    (bytes.fromhex('c3b0c5b8e28099c5a0'), b'&#x1F489;'),   # 💉 syringe
    (bytes.fromhex('c3b0c5b8c2a5e28094'), b'&#x1F955;'),   # 🥕 carrot
    (bytes.fromhex('c3b0c5b8c28fc692'), b'&#x1F3CB;'),     # 🏋 weightlifter
]

for old, new in BYTE_FIXES:
    count = raw.count(old)
    if count > 0:
        print(f'  Replacing {old.hex()} ({count}x) -> {new.decode()}')
        raw = raw.replace(old, new)
    else:
        print(f'  NOT FOUND: {old.hex()}')

# Also replace any orphaned c3b0 c5b8 pairs (broken emoji starts)
orphan = bytes.fromhex('c3b0c5b8')
count = raw.count(orphan)
if count > 0:
    print(f'  Removing {count} orphaned c3b0c5b8 pairs')
    raw = raw.replace(orphan, b'')

# Verify UTF-8
try:
    text = raw.decode('utf-8')
    print(f'UTF-8 decode OK. Chars: {len(text)}')
except UnicodeDecodeError as e:
    print(f'UTF-8 decode FAILED: {e}')
    text = raw.decode('latin-1')

with open('src/web/index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done.')
# Final check
with open('src/web/index.html', 'rb') as f:
    final = f.read()
remaining = sum(1 for i in range(len(final)-1) if final[i] == 0xC3 and final[i+1] == 0xB0)
print(f'Remaining C3B0 sequences: {remaining}')
