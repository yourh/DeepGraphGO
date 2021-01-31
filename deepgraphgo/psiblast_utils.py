#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

from pathlib import Path
from collections import defaultdict
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio.Blast import NCBIXML
from tqdm import tqdm
from logzero import logger

__all__ = ['psiblast', 'blast']


def psiblast(blastdb, pid_list, fasta_path, output_path: Path, evalue=1e-3, num_iterations=3,
             num_threads=40, bits=True, query_self=False, **kwargs):
    output_path = output_path.with_suffix('.xml')
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cline = NcbipsiblastCommandline(query=fasta_path, db=blastdb, evalue=evalue, outfmt=5, out=output_path,
                                        num_iterations=num_iterations, num_threads=num_threads, **kwargs)
        logger.info(cline)
        cline()
    else:
        logger.info(F'Using exists blast output file {output_path}')
    with open(output_path) as fp:
        psiblast_sim = defaultdict(dict)
        for pid, rec in zip(tqdm(pid_list, desc='Parsing PsiBlast results'), NCBIXML.parse(fp)):
            query_pid, sim = rec.query, []
            assert pid == query_pid
            for alignment in rec.alignments:
                alignment_pid = alignment.hit_def.split()[0]
                if alignment_pid != query_pid or query_self:
                    psiblast_sim[query_pid][alignment_pid] = max(hsp.bits if bits else hsp.identities / rec.query_length
                                                                 for hsp in alignment.hsps)
    return psiblast_sim


def blast(*args, **kwargs):
    return psiblast(*args, **kwargs, num_iterations=1)
