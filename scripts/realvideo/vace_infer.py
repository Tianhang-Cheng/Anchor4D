#!/usr/bin/env python3
"""LightX2V `lightx2v.infer` for Wan2.1-VACE with a raised frame cap (run with the LightX2V environment).

Stock LightX2V caps a 480x832 VACE window at 32760 latent patches (81 frames). The completion stage keeps a whole clip
in one window so the model sees the full trajectory; VACE_SEQ_LEN raises the cap (latent frames x 1560 at 480x832).
All arguments are forwarded to lightx2v.infer.
"""
import os
import sys

from lightx2v.models.input_encoders.hf.vace.vace_processor import VaceVideoProcessor

_cap = int(os.environ.get('VACE_SEQ_LEN', '0'))
if _cap:
    VaceVideoProcessor.set_seq_len = lambda self, seq_len, _cap=_cap: setattr(self, 'seq_len', max(seq_len, _cap))

from lightx2v.infer import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
