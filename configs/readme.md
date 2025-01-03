Baseline settings
=================
- baseline_ctc.yaml
Baseline experiment using only CTC loss

- baseline_sot_ctc.yaml
Baseline experiment using SOT together with CTC loss

- baseline_sot_only.yaml
Baseline experiment using SOT without CTC loss


SACTC settings
==============
- proposed_sactc_sot_r05.yaml
Proposed SACTC loss with a risk factor of 5

- proposed_sactc_sot_r10.yaml
Proposed SACTC loss with a risk factor of 10

- proposed_sactc_sot_r15.yaml
Proposed SACTC loss with a risk factor of 15

- proposed_sactc_sot_r20.yaml
Proposed SACTC loss with a risk factor of 20


Decoding settings
=================
- decode/
  - decode_asr_ctc.yaml
  CTC decoding with a beam size of 10

  - decode_asr_aed_only.yaml
  AED decoding with a beam size of 10
 
  - decode_asr_aed_ctc.yaml 
  AED-CTC joint decoding with a beam size of 10

