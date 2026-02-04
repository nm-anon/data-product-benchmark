# Benchmark Characteristics

## ConvFinQA
### Dataset-Level
- Total examples: 3113
- Splits: {'train': 2113, 'dev': 373, 'test': 627}
- Avg DPR length (words): 77.16
- Ground truth refs: tables=14787, text=424596, synth_text=26171
- Unique referenced IDs: tables=4159, text=33629, synth_text=7309
- Topic coverage (top terms):
  - financial (coverage 0.6341)
  - tax (coverage 0.1574)
  - company's (coverage 0.2946)
  - performance (coverage 0.4616)
  - net (coverage 0.2865)
  - stock (coverage 0.2014)
  - impact (coverage 0.5503)
  - cash (coverage 0.177)
  - trends (coverage 0.4507)
  - debt (coverage 0.1503)
### Corpus-Level
- Entries: 4976
- Unique table IDs: 4976
- Text snippets: 114272
- Synth text entries: 8721
- Avg table rows: 2.88
- Avg table cols: 6.4
- Topic coverage (top terms):
  - financial (coverage 0.162)
  - stock (coverage 0.1159)
  - performance (coverage 0.1047)
  - return (coverage 0.0834)
  - december (coverage 0.0831)
  - inc (coverage 0.0872)
  - corporation (coverage 0.0808)
  - total (coverage 0.0653)
  - net (coverage 0.0662)
  - cumulative (coverage 0.0634)

## HybridQA
### Dataset-Level
- Total examples: 8820
- Splits: {'train': 4843, 'dev': 1997, 'test': 1980}
- Avg DPR length (words): 76.41
- Ground truth refs: tables=61889, text=2590116, synth_text=208078
- Unique referenced IDs: tables=11490, text=220500, synth_text=38625
- Topic coverage (top terms):
  - historical (coverage 0.4551)
  - team (coverage 0.209)
  - performance (coverage 0.3607)
  - player (coverage 0.1306)
  - support (coverage 0.5243)
  - details (coverage 0.405)
  - across (coverage 0.4053)
  - queries (coverage 0.4655)
  - events (coverage 0.2561)
  - sports (coverage 0.2011)
### Corpus-Level
- Entries: 12378
- Unique table IDs: 12378
- Text snippets: 413952
- Synth text entries: 41608
- Avg table rows: 15.79
- Avg table cols: 4.49
- Topic coverage (top terms):
  - list (coverage 0.0962)
  - historic (coverage 0.0554)
  - notable (coverage 0.0532)
  - world (coverage 0.0489)
  - national (coverage 0.0435)
  - league (coverage 0.0405)
  - season (coverage 0.037)
  - early (coverage 0.0368)
  - county (coverage 0.0333)
  - five (coverage 0.034)

## TATQA
### Dataset-Level
- Total examples: 1143
- Splits: {'train': 820, 'dev': 147, 'test': 176}
- Avg DPR length (words): 95.28
- Ground truth refs: tables=4404, text=28445, synth_text=7616
- Unique referenced IDs: tables=1497, text=8711, synth_text=2573
- Topic coverage (top terms):
  - financial (coverage 0.6667)
  - tax (coverage 0.2476)
  - revenue (coverage 0.2948)
  - cash (coverage 0.1864)
  - net (coverage 0.4418)
  - assets (coverage 0.2992)
  - income (coverage 0.3342)
  - changes (coverage 0.8294)
  - total (coverage 0.3666)
  - expenses (coverage 0.3053)
### Corpus-Level
- Entries: 2757
- Unique table IDs: 2757
- Text snippets: 13155
- Synth text entries: 4760
- Avg table rows: 2.99
- Avg table cols: 8.37
- Topic coverage (top terms):
  - financial (coverage 0.0987)
  - tax (coverage 0.0595)
  - revenue (coverage 0.0677)
  - income (coverage 0.0597)
  - net (coverage 0.0581)
  - performance (coverage 0.0577)
  - cash (coverage 0.0382)
  - share (coverage 0.0439)
  - expense (coverage 0.045)
  - segment (coverage 0.0367)
