# Q4: data statistics with the T5 tokenizer, before and after pre-processing.
# lengths are in T5 tokens and include the final </s>; vocab size = number of distinct token ids.
from transformers import T5TokenizerFast
from load_data import PREFIX, normalize_times, load_lines

tok = T5TokenizerFast.from_pretrained('google-t5/t5-small')

def stats(nl, sql):
    e = tok(nl)['input_ids']; d = tok(sql)['input_ids']
    return (len(nl), sum(map(len, e)) / len(e), sum(map(len, d)) / len(d),
            len({t for x in e for t in x}), len({t for x in d for t in x}))

def show(title, fn_nl, fn_sql):
    print(f"\n== {title}")
    print(f"{'split':6}{'examples':>10}{'mean NL len':>13}{'mean SQL len':>14}{'NL vocab':>10}{'SQL vocab':>11}")
    for split in ['train', 'dev']:
        nl = [fn_nl(q) for q in load_lines(f'data/{split}.nl')]
        sql = [fn_sql(q) for q in load_lines(f'data/{split}.sql')]
        n, a, b, v1, v2 = stats(nl, sql)
        print(f"{split:6}{n:>10}{a:>13.2f}{b:>14.2f}{v1:>10}{v2:>11}")

ident = lambda x: x
show("Table 1: before pre-processing (raw text)", ident, ident)
show("Table 2a: prefix + less_than", lambda q: PREFIX + q, lambda q: q.replace('<', 'less_than'))
show("Table 2b: prefix + less_than + normalize_times", lambda q: PREFIX + normalize_times(q), lambda q: q.replace('<', 'less_than'))
# how many questions does the time rule touch?
for split in ['train', 'dev']:
    qs = load_lines(f'data/{split}.nl')
    print(f"{split}: normalize_times changed {sum(normalize_times(q) != q for q in qs)}/{len(qs)} questions")
