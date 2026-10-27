"""Ingest TCGA paired RNA + RPPA protein.

Pairing: TCGA barcodes. The Xena RNA matrix (EB++AdjustPANCAN) labels samples
with 15-char barcodes (TCGA-XX-XXXX-01); TCPA Level-4 RPPA rows carry full
44-char barcodes (TCGA-XX-XXXX-01A-21-A17I-20). Therefore, the first 15 chars 
should indicate the same tumor specimen measured by both assays.

Filters (adapted from deepgxp): solid tumor types with > 50 paired samples;
hematopoietic/lymphoid (LAML, DLBC) excluded. Antibody to gene: TCPA uses
the same MD Anderson RPPA naming as CCLE RPPA. Mapped with the curated MANUAL table
and mygene fallback. Unmappable antibodies are dropped and listed.
"""
import io
import time
import os
import pathlib
import re
import urllib.request

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out" / "tcga"
RAW = HERE / "raw"

RNA_URL = ("https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/"
           "download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz")
RNA_FILE = RAW / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
RPPA_URL = "https://ndownloader.figshare.com/files/22215147"
RPPA_FILE = RAW / "TCGA-PANCAN32-L4.csv"

MIN_SAMPLES_PER_TYPE = 50
EXCLUDE_TYPES = {"TCGA-LAML", "TCGA-DLBC"}  # hematopoietic & lymphoid

FILTER_FILE = RAW / 'ccle_filter_antibody_correlations.csv'

MANUAL = {
    '1433BETA': ['YWHAB'], '1433EPSILON': ['YWHAE'], '1433ZETA': ['YWHAZ'],
    'ACETYLATUBULIN': ['TUBA1A'], 'ALPHACATENIN': ['CTNNA1'],
    'AMPKALPHA': ['PRKAA1', 'PRKAA2'], 'ANNEXINVII': ['ANXA7'],
    'BAP1C4': ['BAP1'], 'BECLIN': ['BECN1'], 'BETAACTIN': ['ACTB'],
    'CASPASE7CLEAVEDD198': ['CASP7'], 'CIAP': ['BIRC2'], 'CLAUDIN7': ['CLDN7'],
    'COLLAGENVI': ['COL6A1'], 'CYCLINB1': ['CCNB1'], 'CYCLIND1': ['CCND1'],
    'CYCLINE1': ['CCNE1'], 'CYCLINE2': ['CCNE2'], 'ERALPHA': ['ESR1'],
    'FIBRONECTIN': ['FN1'], 'GSK3': ['GSK3A', 'GSK3B'],
    'GSK3ALPHABETA': ['GSK3A', 'GSK3B'], 'GSK3BETA': ['GSK3B'],
    'GSK3PS9': ['GSK3B'], 'MAPK': ['MAPK1', 'MAPK3'],
    'MAPKPT202Y204': ['MAPK1', 'MAPK3'], 'P38': ['MAPK14'],
    'P38ALPHAMAPK': ['MAPK14'], 'P38MAPK': ['MAPK14'],
    'MDMXMDM4': ['MDM4'], 'MYOSINIIAPS1943': ['MYH9'],
    'NFKBP65PS536': ['RELA'], 'PCADHERIN': ['CDH3'], 'P62LCKLIGAND': ['LCK'],
    'P70S6K': ['RPS6KB1'], 'P70S6KPT389': ['RPS6KB1'],
    'PARPCLEAVED': ['PARP1'], 'PAXILLIN': ['PXN'], 'PKCDELTAPS664': ['PRKCD'],
    'PKCPANBETAII': ['PRKCB'], 'PKCPANBETAIIPS660': ['PRKCB'],
    'RAPTOR': ['RPTOR'], 'RSK123': ['RPS6KA1', 'RPS6KA2', 'RPS6KA3'],
    'STAT5ALPHA': ['STAT5A'], 'STATHMIN': ['STMN1'], 'TAZ': ['WWTR1'],
    'ACC': ['ACACA'], 'ACCS79': ['ACACA'], 'CD20': ['MS4A1'], 'AR': ['AR'],
    'BIM': ['BCL2L11'], 'MRE11': ['MRE11A', 'MRE11'],
    'ANNEXIN1': ['ANXA1'], 'BCLXL': ['BCL2L1'], 'BETACATENIN': ['CTNNB1'],
    'CRAF': ['RAF1'], 'CASPASE8': ['CASP8'], 'CAVEOLIN1': ['CAV1'],
    'ECADHERIN': ['CDH1'], 'HEREGULIN': ['NRG1'], 'NCADHERIN': ['CDH2'],
    'P14ARF': ['CDKN2A'], 'PI3KP110ALPHA': ['PIK3CA'], 'PI3KP85': ['PIK3R1'],
    'SHP2': ['PTPN11'], 'TRANSGLUTAMINASE': ['TGM2'], 'TUBERIN': ['TSC2'],
    'P21': ['CDKN1A'], 'P27': ['CDKN1B'], 'JNK': ['MAPK8', 'MAPK9', 'MAPK10'],
    'P90RSK': ['RPS6KA1', 'RPS6KA2', 'RPS6KA3'],
    # TCPA-panel-specific names
    '4EBP1': ['EIF4EBP1'], '4EBP1PS65': ['EIF4EBP1'],
    '4EBP1PT37T46': ['EIF4EBP1'], '4EBP1PT70': ['EIF4EBP1'],
    '53BP1': ['TP53BP1'], 'CKIT': ['KIT'], 'CMET': ['MET'],
    'CMETPY1235': ['MET'], 'CMYC': ['MYC'], 'P70S6K1': ['RPS6KB1'],
    'MYOSINIIA': ['MYH9'], 'RAB11': ['RAB11A'], 'CABL': ['ABL1'],
    'CASPASE3': ['CASP3'], 'CASPASE9': ['CASP9'],
    'COMPLEXIISUBUNIT30': ['SDHB'], 'OXPHOSCOMPLEXVSUBUNITB': ['ATP5F1A'],
    'GYGGLYCOGENIN1': ['GYG1'], 'HIF1ALPHA': ['HIF1A'], 'LCN2A': ['LCN2'],
    'PARPAB3': ['PARP1'], 'THYMIDILATESYNTHASE': ['TYMS'],
    'CHROMOGRANINANTERM': ['CHGA'], 'NAPSINA': ['NAPSA'],
    'SYNAPTOPHYSIN': ['SYP'], 'PYGBAB2': ['PYGB'],
}


def base_name(ab):
    n = re.sub(r'_Caution$', '', ab)
    n = re.sub(r'_p[STY][0-9]+.*$', '', n, flags=re.I)
    return n.replace('_', ' ').strip()


def norm(s):
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def download(url, dest):
    if dest.exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    print(f'downloading {url} -> {dest}')
    urllib.request.urlretrieve(url, dest)


def map_antibodies(antibodies):
    import requests
    ab2gene = {}
    uniq = sorted({base_name(a) for a in antibodies})
    by_q = {}
    for i in range(0, len(uniq), 50):
        for attempt in range(4):
            try:
                r = requests.post('https://mygene.info/v3/query',
                                  params={'scopes': 'symbol,alias',
                                          'fields': 'symbol', 'species': 'human'},
                                  json={'q': uniq[i:i + 50]}, timeout=60)
                for e in r.json():
                    by_q.setdefault(e['query'], []).append(e)
                break
            except Exception as e:
                print(f'  mygene retry {i}: {str(e)[:80]}')
                time.sleep(5)
    for ab in antibodies:
        bn = re.sub(r'\([^)]*\)', '', base_name(ab)).strip()
        keys = [norm(bn)] + ([norm(bn[1:])] if bn.startswith('X') else [])
        c = next((MANUAL[k] for k in keys if k in MANUAL), None)
        if c:
            ab2gene[ab] = c[0]
            continue
        entries = [e for e in by_q.get(base_name(ab), []) if e.get('symbol')]
        exact = [e['symbol'].upper() for e in entries
                 if norm(e['symbol']) == norm(base_name(ab))]
        ab2gene[ab] = (exact or [entries[0]['symbol'].upper()])[0] if entries else None
    return ab2gene


def tumor_types(barcodes):
    cache = OUT / 'tcga_tumortypes.csv'
    if cache.exists():
        m = pd.read_csv(cache, index_col=0).iloc[:, 0]
        return m.to_dict()
    import json
    import time
    import requests
    patients = sorted({b[:12] for b in barcodes})
    p2t = {}
    for i in range(0, len(patients), 300):
        batch = patients[i:i + 300]
        for attempt in range(4):
            try:
                r = requests.get('https://api.gdc.cancer.gov/v0/cases',
                                 params={'filters': json.dumps(
                                     {'op': 'in', 'content': {
                                         'field': 'submitter_id',
                                         'value': batch}}),
                                     'fields': 'project.project_id,submitter_id',
                                     'size': 3000, 'format': 'JSON'}, timeout=120)
                for hit in r.json()['data']['hits']:
                    p2t[hit['submitter_id']] = hit['project']['project_id']
                break
            except Exception as e:
                print(f'  GDC retry {i}: {str(e)[:80]}')
                time.sleep(5)
        time.sleep(0.2)
    m = {b: p2t.get(b[:12], 'UNKNOWN') for b in barcodes}
    pd.Series(m).to_csv(cache)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # RNA
    download(RNA_URL, RNA_FILE)
    print('loading Xena RNA matrix (large, ~1 min)...')
    rna = pd.read_csv(RNA_FILE, sep='\t', index_col=0)
    rna.columns = [c[:15] for c in rna.columns]
    rna = rna.T.groupby(level=0).mean().T        # collapse duplicate barcodes
    numeric = pd.Series([str(x).isdigit() for x in rna.index], index=rna.index)
    if numeric.any():
        print(f'dropping {int(numeric.sum())} numeric-ID rows (no gene symbol)')
        rna = rna[~pd.Series(numeric, index=rna.index).to_numpy()]
    rna.index = pd.Index([str(g) for g in rna.index], dtype=object)
    rna.index.name = 'gene'
    print(f'RNA: {rna.shape[0]} genes x {rna.shape[1]} samples (symbol-keyed)')

    # protein
    download(RPPA_URL, RPPA_FILE)
    rppa = pd.read_csv(RPPA_FILE, index_col=0)
    rppa.index = [b[:15] for b in rppa.index.astype(str)]
    rppa = rppa[~rppa.index.duplicated(keep='first')]
    rppa = rppa.drop(columns=[c for c in ('Cancer_Type', 'Sample_Type')
                              if c in rppa.columns])
    print(f'RPPA: {rppa.shape[0]} samples x {rppa.shape[1]} antibodies')

    # pair
    shared = rna.columns.intersection(rppa.index)
    print(f'paired samples: {len(shared)}')
    rna, rppa = rna[shared], rppa.loc[shared]

    # tumor-type filter
    tmap = tumor_types(list(shared))
    tt = pd.Series({s: tmap.get(s, 'UNKNOWN') for s in shared})
    keep_types = tt.value_counts()
    keep_types = keep_types[(keep_types > MIN_SAMPLES_PER_TYPE)].index
    keep_types = [t for t in keep_types if t not in EXCLUDE_TYPES]
    mask = tt.isin(keep_types)
    n_before = len(shared)
    rna, rppa, tt = rna.loc[:, mask.values], rppa.loc[mask.values], tt[mask]
    print(f'after tumor-type filter (> {MIN_SAMPLES_PER_TYPE} paired samples, '
          f'no {EXCLUDE_TYPES}): {len(tt)} samples, {len(keep_types)} types')

    # antibody mapping
    ab2gene = map_antibodies(list(rppa.columns))
    mapped = {a: g for a, g in ab2gene.items() if g}
    dropped = [a for a, g in ab2gene.items() if not g]
    print(f'antibodies mapped: {len(mapped)}/{len(ab2gene)}; dropped: {dropped}')
    prot = rppa[list(mapped)].copy()
    prot.columns = [mapped[a] for a in prot.columns]
    prot = prot.T.groupby(level=0).mean().T

    if FILTER_FILE.exists():
        filt = pd.read_csv(FILTER_FILE).dropna(subset=['r'])
        passed = filt[(filt.r >= 0.3) & (filt.p <= 0.05)]
        valid_genes = set(passed.gene)
        n_before = prot.shape[1]
        prot = prot[[g for g in prot.columns if g in valid_genes]]
        print(f'CCLE-validated protein filter: kept '
              f'{prot.shape[1]}/{n_before} genes '
              f'({len(valid_genes)} validated genes total)')
    else:
        print(f'{FILTER_FILE.name} not found. Skip CCLE '
              'protein filter')
    prot = prot.T          
    prot.index.name = 'gene'

    rna.to_csv(OUT / 'tcga_rna.csv.gz', compression='gzip')
    prot.to_csv(OUT / 'tcga_protein.csv.gz', compression='gzip')
    pd.DataFrame({'sample_id': tt.index, 'tumor_type': tt.values}).to_csv(
        OUT / 'tcga_samples.csv', index=False)
    print(f'wrote {OUT}: RNA {rna.shape}, protein {prot.shape}')


if __name__ == '__main__':
    main()
