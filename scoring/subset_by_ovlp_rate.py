 #!/usr/bin/env python

"""
Author: Jiawen Kang (jwkang at se.cuhk.edu.hk)

This script is used to subset the ref and hyp trn files by the ovlp rate.
It requires a utt2rate file to specify the ovlp rate of each utt.
"""

import os
import click


@click.command()
@click.option("--utt2rate", type=click.Path(exist=True), required=True, help="Utt2ovlp file")
@click.option("--ref", type=click.Path(exist=True), required=True, help="Reference trn")
@click.option("--hyp", type=click.Path(exist=True), required=True, help="Hypothsis trn")
@click.option("--out_dir", type=click.Path(), required=True, help="Output directory")
def main(
    utt2rate: click.Path,
    ref: click.Path,
    hyp: click.Path,
    out_dir: click.Path
):

    # 1. group utts by ovlp rate
    part1_name_list = []
    part2_name_list = []
    part3_name_list = []
    with open(utt2rate, 'r') as f:
        for line in f.readlines():
            name, rate = line.strip().split()
            if float(rate) <= 0.20:
                part1_name_list.append(name)
            elif float(rate) <= 0.50:
                part2_name_list.append(name)
            elif float(rate) <= 1.00:
                part3_name_list.append(name)
            else:
                raise KeyError

    # 2. subset
    f_ref1 = open(os.path.join(out_dir, 'ref_0.20.trn'), 'w')
    f_ref2 = open(os.path.join(out_dir, 'ref_0.50.trn'), 'w')
    f_ref3 = open(os.path.join(out_dir, 'ref_1.00.trn'), 'w')

    f_hyp1 = open(os.path.join(out_dir, 'hyp_0.20.trn'), 'w')
    f_hyp2 = open(os.path.join(out_dir, 'hyp_0.50.trn'), 'w')
    f_hyp3 = open(os.path.join(out_dir, 'hyp_1.00.trn'), 'w')

    # 3. write 
    with open(ref, 'r') as ref:
        for line in ref.readlines():
            name = line.split('(')[1].split(')')[0]
            name = f"{name.split('-')[1]}-{name.split('-')[2]}"
            if name in part1_name_list:
                f_ref1.write(line)
            elif name in part2_name_list:
                f_ref2.write(line)
            elif name in part3_name_list:
                f_ref3.write(line)
            else:
                raise KeyError(part1_name_list, name)

    with open(hyp, 'r') as hyp:
        for line in hyp.readlines():
            name = line.split('(')[1].split(')')[0]
            name = f"{name.split('-')[1]}-{name.split('-')[2]}"
            if name in part1_name_list:
                f_hyp1.write(line)
            elif name in part2_name_list:
                f_hyp2.write(line)
            elif name in part3_name_list:
                f_hyp3.write(line)
            else:
                raise KeyError(name)


if __name__ == '__main__':
    main()
