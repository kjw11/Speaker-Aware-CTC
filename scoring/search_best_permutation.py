#!/usr/bin/env python

"""
Author: Jiawen Kang (jwkang at se.cuhk.edu.hk)

This script is used to assist permutation invariant scoring, it split 
multi-talker ASR ref. and hyp. trn files into utterance-wise files.
Each utterance contains sub-directories for different permutations. 
In ref.trn, text from different speakers are divided by a '$' sign, e.g.:
"A B C D $ E F G\t(1089-1089-test_clean_2mix_0000)"
where the possible permutations are "A B C D $ E F G" and "E F G $ A B C D".

Expected usage:
python3 splitting_trn.py \
    		--ref_utt $ref_utt \
    		--hyp $hyp \
    		--out_dir $out_dir

output dir layouts:
- out_dir:
    	- tmp:
    	|	- utt0:
    	|	|	- permutation0
    	|	|	|	- ref.trn
    	|	|	|	- hyp.trn
    	|	|	- permutation1
    	|	|	|	...
    	|	- utt1:
    	|	|	...

Notes:
1. There will be two possible permutation in 2-speaker case, and 6 in 3-speaker case.
2. This script uses @click to parse arguments, please install it before running.
"""

import os
import click
import tqdm
import editdistance

from itertools import permutations

def read_trn(trn):
    with open(trn, 'r') as f:
        lines = f.readlines()
    return lines

def compute_wer(ref, hyp):
    ref = ref.strip().split(' ')
    hyp = hyp.strip().split(' ')
    return editdistance.eval(ref, hyp) / len(ref)

@click.command()
@click.option('--ref_utt', type=str, default='n/a')
@click.option('--hyp', type=str, default='n/a')
@click.option('--out_dir', type=str, default='n/a')
def main(ref_utt, hyp, out_dir):
    # 1. read trn
    ref_lines = read_trn(ref_utt)
    hyp_lines = read_trn(hyp)
    assert len(ref_lines) == len(hyp_lines), "ref and hyp have different number of lines"

    # 2. split ref trn
    best_refs = []
    best_hyps = []
    for line_idx, line in tqdm.tqdm(enumerate(ref_lines), mininterval=0.01):
        utt_name = line.split('(')[-1].split(')')[0]
        utt_dir = os.path.join(out_dir, 'tmp', utt_name)

        assert not os.path.exists(utt_dir)
        os.makedirs(utt_dir)

        spkr_list = line.strip().split('\t')[0].split(' $ ')
        assert len(spkr_list) == 2 or len(spkr_list) == 3, "only support 2 or 3 speakers"

        # find all permutations
        best_ref = None
        best_wer = 99
        for permu_idx, permu in enumerate(permutations(spkr_list)):
            ref = ' '.join(permu)
            hyp = hyp_lines[line_idx].strip().split('\t')[0].replace(' $ ', ' ')
            wer = compute_wer(ref, hyp)

            if wer <= best_wer:
                best_wer = wer
                best_ref = ref

        assert utt_name in hyp_lines[line_idx], "utterance name mismatch"
        best_refs.append(best_ref + '\t(' + utt_name + ')')
        best_hyps.append(hyp_lines[line_idx].strip().replace(' $ ', ' '))

    # 3. write best ref and hyp
    with open(os.path.join(out_dir, 'best_ref.trn'), 'w') as f:
        f.write('\n'.join(best_refs))
    with open(os.path.join(out_dir, 'best_hyp.trn'), 'w') as f:
        f.write('\n'.join(best_hyps))


if __name__ == '__main__':
    main()
