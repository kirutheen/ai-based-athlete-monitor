with open('src/web/index.html', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines, 1):
    if 'dof-card' in l or 'dof-sub' in l or 'dof-val' in l or 'dof-title' in l:
        print(i, l.rstrip().encode('ascii', errors='replace').decode('ascii'))
