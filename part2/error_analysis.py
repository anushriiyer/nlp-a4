# Q6 / Table 5: qualitative error analysis of the best model on the dev set (regex-based; classes overlap)
import pickle, re, sys, collections
stem = sys.argv[1] if len(sys.argv) > 1 else 'ft_lr3e4'   # e.g. ft_lr3e4 or scr_scr_long_lr3e4
gold = [l.strip() for l in open('data/dev.sql')]
pred = [l.rstrip('\n') for l in open(f'results/t5_{stem}_dev.sql')]
grec, _ = pickle.load(open('records/ground_truth_dev.pkl', 'rb'))
prec, perr = pickle.load(open(f'records/t5_{stem}_dev.pkl', 'rb'))
N = len(gold)
wrong = [i for i in range(N) if set(grec[i]) != set(prec[i])]
print(f"dev examples {N}; record-incorrect {len(wrong)} ({len(wrong)/N:.1%})")

def consts(q):  return sorted(re.findall(r"'[^']*'", q))
def nums(q):    return sorted(re.findall(r"(?<![\w.'])\d+(?![\w.'])", q.replace('1 = 1', '')))  # ignore the 'AND 1 = 1' filler
def tables(q):
    m = re.search(r'FROM (.*?) WHERE', q)
    return sorted({t.split()[0] for t in m.group(1).split(',')}) if m else []
def select(q):  return re.sub(r'\s+', ' ', q.split(' FROM ')[0])
def aggs(q):    return sorted(re.findall(r'\b(MIN|MAX|COUNT|SUM|AVG)\s*\(', q))

classes = collections.OrderedDict((k, []) for k in
    ['invalid SQL (database error)', 'wrong quoted constant', 'wrong number (time/date/...)',
     'wrong or missing table', 'wrong SELECT clause', 'wrong aggregate (MIN/MAX/COUNT)'])
other = []
for i in wrong:
    g, p = gold[i], pred[i]
    hit = False
    if perr[i]:
        classes['invalid SQL (database error)'].append(i); hit = True
        # structural comparisons below are meaningless for broken queries
    else:
        if consts(g) != consts(p): classes['wrong quoted constant'].append(i); hit = True
        if nums(g) != nums(p): classes['wrong number (time/date/...)'].append(i); hit = True
        if tables(g) != tables(p): classes['wrong or missing table'].append(i); hit = True
        if select(g) != select(p): classes['wrong SELECT clause'].append(i); hit = True
        if aggs(g) != aggs(p): classes['wrong aggregate (MIN/MAX/COUNT)'].append(i); hit = True
    if not hit: other.append(i)

print(f"\n{'class':38}{'count/dev':>12}{'count/incorrect':>18}")
for k, v in classes.items():
    print(f"{k:38}{len(v):>6}/{N:<5}{len(v):>9}/{len(wrong)}")
print(f"{'none of the above (other)':38}{len(other):>6}/{N:<5}{len(other):>9}/{len(wrong)}")

print("\nerror-message kinds among invalid SQL:")
c = collections.Counter(re.sub(r'".*?"', '"..."', perr[i])[:40] for i in classes['invalid SQL (database error)'])
for k, n in c.most_common(6): print(f"  {n:3} {k}")

nl = [l.strip() for l in open('data/dev.nl')]
def diff(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return sorted((ca - cb).elements()), sorted((cb - ca).elements())
def show(k, ids, n=4):
    print(f"\n### {k}")
    for i in ids[:n]:
        print(f"   consts gold/pred diff: {diff(consts(gold[i]), consts(pred[i]))}  nums: {diff(nums(gold[i]), nums(pred[i]))}  tables: {diff(tables(gold[i]), tables(pred[i]))}")
        print(f"[{i}] NL  : {nl[i]}\n     GOLD: {gold[i][:260]}\n     PRED: {pred[i][:260]}\n     ERR : {perr[i][:80]}")
for k, v in classes.items(): show(k, v)
