with open('src/web/index.html', 'rb') as f:
    raw = f.read()

deg_count = raw.count(b'\xc2\xb0')
print('Degree symbol count (correct C2 B0):', deg_count)

repl_count = raw.count(b'\xef\xbf\xbd')
print('Replacement char count (EF BF BD):', repl_count)

positions = [i for i in range(len(raw)-2) if raw[i:i+3] == b'\xef\xbf\xbd']
for pos in positions[:10]:
    ctx = raw[max(0,pos-20):pos+25]
    decoded = ctx.decode('utf-8', errors='replace')
    print(f'  pos={pos}: ' + repr(decoded))
