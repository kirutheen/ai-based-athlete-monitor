"""
Fix remaining garbled emoji in index.html.
Strategy: read the file as bytes, find all double-encoded 4-byte emoji sequences,
and replace them with HTML entities.

A 4-byte emoji F0 9F XX YY gets double-encoded as:
F0 -> C3 B0 (UTF-8 of U+00F0)
9F -> C5 B8 (wait, 9F = U+009F, which in latin-1 is C2 9F)
Actually: each byte B is encoded as its UTF-8 latin-1 codepoint:
  B < 0x80: stays as-is
  0x80 <= B <= 0xBF: becomes C2 (B) [two bytes]
  0xC0 <= B <= 0xFF: becomes C3 (B-0x40) [two bytes]

So F0 9F XX YY becomes:
  F0 -> C3 B0
  9F -> C2 9F  
  XX -> depends on XX
  YY -> depends on YY
"""
import re
import struct

def latin1_byte_to_utf8(b):
    """How a latin-1 byte gets re-encoded as UTF-8."""
    if b < 0x80:
        return bytes([b])
    else:
        # Encode as UTF-8 the latin-1 codepoint
        return bytes([0xC0 | (b >> 6), 0x80 | (b & 0x3F)])

def decode_double_encoded(raw, pos):
    """Try to decode a double-encoded sequence starting at pos.
    Returns (original_bytes, new_pos) or None if not a double-encoded sequence."""
    result = bytearray()
    i = pos
    while i < len(raw):
        b = raw[i]
        if b < 0x80:
            # Single byte - was originally < 0x80
            result.append(b)
            i += 1
        elif b == 0xC2 and i+1 < len(raw):
            # Was originally a byte in 0x80-0xBF
            orig = raw[i+1]
            if 0x80 <= orig <= 0xBF:
                result.append(orig)
                i += 2
            else:
                break
        elif b == 0xC3 and i+1 < len(raw):
            # Was originally a byte in 0xC0-0xFF
            orig = raw[i+1] + 0x40
            result.append(orig)
            i += 2
        elif b == 0xC5 and i+1 < len(raw) and raw[i+1] in [0xB8, 0xB9, 0xBE, 0xBF]:
            # C5 B8 = U+0178 = latin1 0x9F, C5 B9 = not used, etc.
            # C5 B8 -> original byte 0x9F
            if raw[i+1] == 0xB8:
                result.append(0x9F)
            else:
                break
            i += 2
        else:
            break
        
        # Check if we have a complete UTF-8 sequence
        try:
            decoded = result.decode('utf-8')
            # If all chars are in emoji range, we're done
            if all(ord(c) > 0x7F for c in decoded):
                return bytes(result), i
        except UnicodeDecodeError:
            if len(result) > 6:  # Too long, give up
                break
    return None

with open('src/web/index.html', 'rb') as f:
    raw = f.read()

print(f'File size: {len(raw)}')

# Find all double-encoded emoji starts (C3 B0 = double-encoded F0)
positions = []
i = 0
while i < len(raw) - 1:
    if raw[i] == 0xC3 and raw[i+1] == 0xB0:
        positions.append(i)
    i += 1

print(f'Found {len(positions)} double-encoded sequences at: {positions}')

# Read context around each to identify what emoji it should be
for pos in positions:
    # Extract up to 16 bytes and try to decode
    chunk = raw[pos:pos+16]
    print(f'  pos={pos}: {chunk.hex()} | latin1: {chunk.decode("latin-1")!r}')
    
    # Try to decode: C3B0 = F0, then continue...
    # Next bytes tell us the emoji codepoint
    # Build original bytes
    orig_bytes = bytearray()
    i2 = pos
    while i2 < min(pos+12, len(raw)):
        b = raw[i2]
        if b == 0xC3 and i2+1 < len(raw):
            orig_bytes.append(raw[i2+1] + 0x40)
            i2 += 2
        elif b == 0xC2 and i2+1 < len(raw):
            orig_bytes.append(raw[i2+1])
            i2 += 2
        elif b == 0xC5 and i2+1 < len(raw) and raw[i2+1] == 0xB8:
            orig_bytes.append(0x9F)
            i2 += 2
        elif b == 0xE2 and i2+2 < len(raw):
            # This is a real UTF-8 3-byte char, stop here
            break
        elif b < 0x80:
            break
        else:
            break
    
    if len(orig_bytes) >= 4:
        try:
            emoji = orig_bytes[:4].decode('utf-8')
            print(f'    -> U+{ord(emoji):04X} = {emoji!r} = {emoji}')
        except Exception as e:
            print(f'    -> decode error: {e} orig={orig_bytes[:4].hex()}')
    else:
        print(f'    -> too short: {orig_bytes.hex()}')
