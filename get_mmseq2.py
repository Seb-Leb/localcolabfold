import os
import re
import sys
import pickle
import pairmsa
import json

import colabfold as cf

from alphafold.data import parsers

import matplotlib.pyplot as plt
import numpy as np

from Bio import SeqIO

import argparse
argparser = argparse.ArgumentParser(description='run alphafold.')
argparser.add_argument('fasta_file', type=str)
args = argparser.parse_args()
fasta_file = args.fasta_file

def get_msa(sequence, jobname):

  sequence = re.sub("[^A-Z:/]", "", sequence.upper())
  sequence = re.sub(":+",":",sequence)
  sequence = re.sub("/+","/",sequence)

  jobname = re.sub(r'\W+', '', jobname)

  # define number of copies
  homooligomer =  "2"
  homooligomer = re.sub("[:/]+",":",homooligomer)
  if len(homooligomer) == 0: homooligomer = "1"
  homooligomer = re.sub("[^0-9:]", "", homooligomer)
  homooligomers = [int(h) for h in homooligomer.split(":")]

  #@markdown - `sequence` Specify protein sequence to be modelled.
  #@markdown  - Use `/` to specify intra-protein chainbreaks (for trimming regions within protein).
  #@markdown  - Use `:` to specify inter-protein chainbreaks (for modeling protein-protein hetero-complexes).
  #@markdown  - For example, sequence `AC/DE:FGH` will be modelled as polypeptides: `AC`, `DE` and `FGH`. A seperate MSA will be generates for `ACDE` and `FGH`.
  #@markdown    If `pair_msa` is enabled, `ACDE`'s MSA will be paired with `FGH`'s MSA.
  #@markdown - `homooligomer` Define number of copies in a homo-oligomeric assembly.
  #@markdown  - Use `:` to specify different homooligomeric state (copy numer) for each component of the complex.
  #@markdown  - For example, **sequence:**`ABC:DEF`, **homooligomer:** `2:1`, the first protein `ABC` will be modeled as a homodimer (2 copies) and second `DEF` a monomer (1 copy).

  ori_sequence = sequence
  sequence = sequence.replace("/","").replace(":","")
  seqs = ori_sequence.replace("/","").split(":")

  if len(seqs) != len(homooligomers):
    if len(homooligomers) == 1:
      homooligomers = [homooligomers[0]] * len(seqs)
      homooligomer = ":".join([str(h) for h in homooligomers])
    else:
      while len(seqs) > len(homooligomers):
        homooligomers.append(1)
      homooligomers = homooligomers[:len(seqs)]
      homooligomer = ":".join([str(h) for h in homooligomers])
      print("WARNING: Mismatch between number of breaks ':' in 'sequence' and 'homooligomer' definition")

  full_sequence = "".join([s*h for s,h in zip(seqs,homooligomers)])

  # prediction directory
  output_dir = 'outputs/prediction_' + jobname #+ '_' + cf.get_hash(full_sequence)[:5]
  os.makedirs(output_dir, exist_ok=True)
  # delete existing files in working directory
  for f in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, f))

  MIN_SEQUENCE_LENGTH = 16
  MAX_SEQUENCE_LENGTH = 2500

  aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
  if not set(full_sequence).issubset(aatypes):
    raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
  if len(full_sequence) < MIN_SEQUENCE_LENGTH:
    raise Exception(f'Input sequence is too short: {len(full_sequence)} amino acids, while the minimum is {MIN_SEQUENCE_LENGTH}')
  if len(full_sequence) > MAX_SEQUENCE_LENGTH:
    raise Exception(f'Input sequence is too long: {len(full_sequence)} amino acids, while the maximum is {MAX_SEQUENCE_LENGTH}. Please use the full AlphaFold system for long sequences.')

  if len(full_sequence) > 1400:
    print(f"WARNING: For a typical Google-Colab-GPU (16G) session, the max total length is ~1400 residues. You are at {len(full_sequence)}! Run Alphafold may crash.")

  print(f"homooligomer: '{homooligomer}'")
  print(f"total_length: '{len(full_sequence)}'")
  print(f"working_directory: '{output_dir}'")
  #%%
  TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

  msa_method = "mmseqs2" #@param ["mmseqs2","jackhmmer","single_sequence","precomputed"]
  pair_msa = False #@param {type:"boolean"}
  pair_cov = 50 #@param [50,75,90] {type:"raw"}
  pair_qid = 20 #@param [15,20,30,40,50] {type:"raw"}
  include_unpaired_msa = True #@param {type:"boolean"}

  add_custom_msa = False #@param {type:"boolean"}
  msa_format = "fas" #@param ["fas","a2m","a3m","sto","psi","clu"]

  # --- Search against genetic databases ---
  os.makedirs('tmp', exist_ok=True)
  msas, deletion_matrices = [],[]

  if add_custom_msa:
    print(f"upload custom msa in '{msa_format}' format")
    msa_dict = files.upload()
    lines = msa_dict[list(msa_dict.keys())[0]].decode()

    # convert to a3m
    with open(f"tmp/upload.{msa_format}","w") as tmp_upload:
      tmp_upload.write(lines)
    os.system(f"reformat.pl {msa_format} a3m tmp/upload.{msa_format} tmp/upload.a3m")
    a3m_lines = open("tmp/upload.a3m","r").read()

    # parse
    msa, mtx = parsers.parse_a3m(a3m_lines)
    msas.append(msa)
    deletion_matrices.append(mtx)

    if len(msas[0][0]) != len(sequence):
      raise ValueError("ERROR: the length of msa does not match input sequence")

  if msa_method == "precomputed":
    print("upload precomputed pickled msa from previous run")
    pickled_msa_dict = files.upload()
    msas_dict = pickle.loads(pickled_msa_dict[list(pickled_msa_dict.keys())[0]])
    msas, deletion_matrices = (msas_dict[k] for k in ['msas', 'deletion_matrices'])

  elif msa_method == "single_sequence":
    if len(msas) == 0:
      msas.append([sequence])
      deletion_matrices.append([[0]*len(sequence)])

  else:
    seqs = ori_sequence.replace('/','').split(':')
    _blank_seq = ["-" * len(seq) for seq in seqs]
    _blank_mtx = [[0] * len(seq) for seq in seqs]
    def _pad(ns,vals,mode):
      if mode == "seq": _blank = _blank_seq.copy()
      if mode == "mtx": _blank = _blank_mtx.copy()
      if isinstance(ns, list):
        for n,val in zip(ns,vals): _blank[n] = val
      else: _blank[ns] = vals
      if mode == "seq": return "".join(_blank)
      if mode == "mtx": return sum(_blank,[])

    if not pair_msa or (pair_msa and include_unpaired_msa):
      # gather msas
      if msa_method == "mmseqs2":
        prefix = cf.get_hash("".join(seqs))
        prefix = os.path.join('tmp',prefix)
        print(f"running mmseqs2")
        A3M_LINES = cf.run_mmseqs2(seqs, prefix, filter=True)

      for n, seq in enumerate(seqs):
        # tmp directory
        prefix = cf.get_hash(seq)
        prefix = os.path.join('tmp',prefix)

        if msa_method == "mmseqs2":
          # run mmseqs2
          a3m_lines = A3M_LINES[n]
          msa, mtx = parsers.parse_a3m(a3m_lines)
          msas_, mtxs_ = [msa],[mtx]

        # pad sequences
        for msa_,mtx_ in zip(msas_,mtxs_):
          msa,mtx = [sequence],[[0]*len(sequence)]
          for s,m in zip(msa_,mtx_):
            msa.append(_pad(n,s,"seq"))
            mtx.append(_pad(n,m,"mtx"))

          msas.append(msa)
          deletion_matrices.append(mtx)

    ####################################################################################
    # PAIR_MSA
    ####################################################################################

    if pair_msa and len(seqs) > 1:
      print("attempting to pair some sequences...")

      if msa_method == "mmseqs2":
        prefix = cf.get_hash("".join(seqs))
        prefix = os.path.join('tmp',prefix)
        print(f"running mmseqs2_noenv_nofilter on all seqs")
        A3M_LINES = cf.run_mmseqs2(seqs, prefix, use_env=False, filter=False)

      _data = []
      for a in range(len(seqs)):
        print(f"prepping seq_{a}")
        _seq = seqs[a]
        _prefix = os.path.join('tmp',cf.get_hash(_seq))

        if msa_method == "mmseqs2":
          a3m_lines = A3M_LINES[a]
          _msa, _mtx, _lab = pairmsa.parse_a3m(a3m_lines,
                                              filter_qid=pair_qid/100,
                                              filter_cov=pair_cov/100)

        elif msa_method == "jackhmmer":
          _msas, _mtxs, _names = run_jackhmmer(_seq, _prefix)
          _msa, _mtx, _lab = pairmsa.get_uni_jackhmmer(_msas[0], _mtxs[0], _names[0],
                                                      filter_qid=pair_qid/100,
                                                      filter_cov=pair_cov/100)

        if len(_msa) > 1:
          _data.append(pairmsa.hash_it(_msa, _lab, _mtx, call_uniprot=False))
        else:
          _data.append(None)

      Ln = len(seqs)
      O = [[None for _ in seqs] for _ in seqs]
      for a in range(Ln):
        if _data[a] is not None:
          for b in range(a+1,Ln):
            if _data[b] is not None:
              print(f"attempting pairwise stitch for {a} {b}")
              O[a][b] = pairmsa._stitch(_data[a],_data[b])
              _seq_a, _seq_b, _mtx_a, _mtx_b = (*O[a][b]["seq"],*O[a][b]["mtx"])
              print(f"found {len(_seq_a)} pairs")
              if len(_seq_a) > 0:
                msa,mtx = [sequence],[[0]*len(sequence)]
                for s_a,s_b,m_a,m_b in zip(_seq_a, _seq_b, _mtx_a, _mtx_b):
                  msa.append(_pad([a,b],[s_a,s_b],"seq"))
                  mtx.append(_pad([a,b],[m_a,m_b],"mtx"))
                msas.append(msa)
                deletion_matrices.append(mtx)


  ####################################################################################
  ####################################################################################

  # save MSA as pickle
  pickle.dump({"msas":msas,"deletion_matrices":deletion_matrices},
              open(os.path.join(output_dir,"msa.pickle"),"wb"))
  pickle.dump({"seqs":seqs, "homooligomers":homooligomers, 'full_sequence':full_sequence, 'ori_sequence':ori_sequence},
              open(os.path.join(output_dir,"seqs_oligos.pickle"),"wb"))

  #########################################
  # Merge and filter
  #########################################
  msa_merged = sum(msas,[])
  if len(msa_merged) > 1:
    print(f'{len(msa_merged)} Sequences Found in Total')
    '''
    if pair_msa:
      ok = {0:True}
      print("running mmseqs2 to merge and filter (-id90) the MSA")
      with open("tmp/raw.fas","w") as fas:
        for n,seq in enumerate(msa_merged):
          seq_unalign = seq.replace("-","")
          fas.write(f">{n}\n{seq_unalign}\n")
      os.system("mmseqs easy-linclust tmp/raw.fas tmp/clu tmp/mmseqs/tmp -c 0.9 --cov-mode 1 --min-seq-id 0.9 --kmer-per-seq-scale 0.5 --kmer-per-seq 80")
      for line in open("tmp/clu_cluster.tsv","r"):
        ok[int(line.split()[0])] = True
      print(f'{len(ok)} Sequences Found in Total (after filtering)')
    else:
    '''
    ok = dict.fromkeys(range(len(msa_merged)),True)

    Ln = np.cumsum(np.append(0,[len(seq) for seq in seqs]))
    Nn,lines = [],[]
    n,new_msas,new_mtxs = 0,[],[]
    for msa,mtx in zip(msas,deletion_matrices):
      new_msa,new_mtx = [],[]
      for s,m in zip(msa,mtx):
        if n in ok:
          new_msa.append(s)
          new_mtx.append(m)
        n += 1
      if len(new_msa) > 0:
        new_msas.append(new_msa)
        new_mtxs.append(new_mtx)
        Nn.append(len(new_msa))
        msa_ = np.asarray([list(seq) for seq in new_msa])
        gap_ = msa_ != "-"
        qid_ = msa_ == np.array(list(sequence))
        gapid = np.stack([gap_[:,Ln[i]:Ln[i+1]].max(-1) for i in range(len(seqs))],-1)
        seqid = np.stack([qid_[:,Ln[i]:Ln[i+1]].mean(-1) for i in range(len(seqs))],-1).sum(-1) / gapid.sum(-1)
        non_gaps = gap_.astype(np.float)
        non_gaps[non_gaps == 0] = np.nan
        lines.append(non_gaps[seqid.argsort()]*seqid[seqid.argsort(),None])

    msas = new_msas
    deletion_matrices = new_mtxs

    Nn = np.cumsum(np.append(0,Nn))

    #########################################
    # Display
    #########################################

    lines = np.concatenate(lines,0)
    if len(lines) > 1:
      plt.figure(figsize=(8,5),dpi=100)
      plt.title("Sequence coverage")
      plt.imshow(lines,
                interpolation='nearest', aspect='auto',
                cmap="rainbow_r", vmin=0, vmax=1, origin='lower',
                extent=(0, lines.shape[1], 0, lines.shape[0]))
      for i in Ln[1:-1]:
        plt.plot([i,i],[0,lines.shape[0]],color="black")

      for j in Nn[1:-1]:
        plt.plot([0,lines.shape[1]],[j,j],color="black")

      plt.plot((np.isnan(lines) == False).sum(0), color='black')
      plt.xlim(0,lines.shape[1])
      plt.ylim(0,lines.shape[0])
      plt.colorbar(label="Sequence identity to query",)
      plt.xlabel("Positions")
      plt.ylabel("Sequences")
      plt.savefig(os.path.join(output_dir,"msa_coverage.png"), bbox_inches = 'tight', dpi=200)
      # plt.show()

for record in SeqIO.parse(fasta_file, 'fasta'):
  jobname = record.description
  sequence = str(record.seq)
  get_msa(sequence, jobname)