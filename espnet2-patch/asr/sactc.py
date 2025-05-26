"""
This code is revised from the open-source implementation of BAYES RISK CTC:
BAYES RISK CTC: CONTROLLABLE CTC ALIGNMENT IN SEQUENCE-TO-SEQUENCE TASKS
"""

import torch
import _k2
import k2
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple


# Large beam size to keep all states in lattices
BEAM_SIZE=1e10

class BPECTC(torch.nn.Module):
    """Base CTC implementation with BPE tokenization.
    
    Args:
        odim (int): Output dimension size (vocab size)
        eprojs (int): Encoder projection size
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0
        reduce (bool, optional): Whether to reduce loss. Defaults to True
        log_semiring (bool, optional): Use log semiring. Defaults to True
    """
    def __init__(
        self, 
        odim: int, 
        eprojs: int, 
        dropout_rate: float = 0.0, 
        reduce: bool = True,
        log_semiring: bool = True,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.probs = None
        self.reduce = reduce
        self.log_semiring = log_semiring

    def forward(
        self, 
        hs_pad: torch.Tensor, 
        hlens: torch.Tensor, 
        ctc_graphs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass computing CTC loss.

        Args:
            hs_pad (torch.Tensor): Encoder output [B, T, D]
            hlens (torch.Tensor): Encoder output lengths [B]
            ctc_graphs (List[torch.Tensor]): CTC topology graphs

        Returns:
            torch.Tensor: CTC loss value
        """
        batch_size = hs_pad.size(0)

        # Build supervision tensor for K2
        supervision = torch.stack([
            torch.arange(batch_size), 
            torch.zeros(batch_size), 
            hlens.cpu()
        ], dim=1).int()
        indices = torch.argsort(supervision[:, 2], descending=True)
        supervision = supervision[indices]

        # Compute log probabilities
        nnet_output = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        nnet_log_prob = F.log_softmax(nnet_output, dim=-1)

        # Create K2 FSA
        dense_fsa_vec = k2.DenseFsaVec(nnet_log_prob, supervision)
        ctc_graphs = self.revise_ctc_graphs(ctc_graphs, indices).requires_grad_(False)
 
        # Intersect dense FSA with CTC graph
        lats = k2.intersect_dense(ctc_graphs, dense_fsa_vec, BEAM_SIZE)
        forward_scores = - lats.get_tot_scores(
            use_double_scores=True, 
            log_semiring=self.log_semiring
        )

        if self.reduce:
            forward_scores = forward_scores.mean()
        else:
            forward_scores = forward_scores.sum()

        # Chache outputs
        self.probs = F.softmax(nnet_output, dim=-1)
        self.loss = forward_scores 
        return forward_scores

    def revise_ctc_graphs(self, mats: List[torch.Tensor]) -> k2.Fsa:
        """Revise CTC graphs to be compatible with standard CTC implementation.
        
        Args:
            mats: List of CTC graph matrices
            
        Returns:
            k2.Fsa: Combined FSA vector
        
        Assumptions:
            1. 'sil' phone is never used (sil_prob always 0)
            2. After removing 'sil' from ilabel table, remaining units match char_list
        """
        ctc_graphs = []
        
        for mat in mats:
            assert not torch.any(mat[:, 2] == 1), "No sil should be used in CTC graph"
            # Adjust indices after removing silence token
            mat[:, 2] = torch.where(mat[:, 2] <= 0, mat[:, 2], mat[:, 2] - 1) 
            ctc_graph = k2.Fsa.from_dict({"arcs": mat.detach()})
            ctc_graphs.append(ctc_graph)

        return k2.create_fsa_vec(ctc_graphs)
        
    def softmax(self, hs_pad):
        self.probs = F.softmax(self.ctc_lo(hs_pad), dim=-1)
        return self.probs

    def log_softmax(self, hs_pad):
        return F.log_softmax(self.ctc_lo(hs_pad), dim=-1)

    def argmax(self, hs_pad):
        return torch.argmax(self.ctc_lo(hs_pad), dim=-1)


class SpeakerAwareCTC(BPECTC):
    """Speaker-aware CTC implementation with Bayesian CTC framework."""
    def __init__(
        self,
        odim: int,
        eprojs: int,
        dropout_rate: float = 0.1,
        reduce: bool = True,
        log_semiring: bool = True,
        risk_strategy: str = "time",
        risk_factor: float = 0.0,
        den_scale: float = 0.0,
        constrain_y: bool = None,
        ignore_nan_grad: bool = False,
    ):
        super().__init__(odim, eprojs, dropout_rate, reduce, log_semiring)
                
        # Map risk strategy with correspond aggregation function
        aggregation_dict = {
            'logistic': 'time',
        }

        # Validate input parameters
        if risk_strategy == "none":
            assert risk_factor == 0.0
        assert 0.0 <= den_scale <= 1.0
        assert risk_factor >= 0.0

        self.strategy = risk_strategy
        self.risk_factor = risk_factor
        self.aggregate = aggregation_dict[self.strategy]
        self.den_scale = den_scale 
        self.target_y = constrain_y
        self.sigmoid = torch.nn.Sigmoid()

        logging.info(f"BRCTC risk strategy: {self.strategy}")
        logging.info(f"BRCTC risk factor: {self.risk_factor}")
        logging.info(f"BRCTC aggregate: {self.aggregate}")
   

    def forward(
        self, 
        hs_pad: torch.Tensor, 
        hlens: torch.Tensor, 
        ys_pad: torch.Tensor, 
        ali: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing SACTC loss.
        
        Args:
            hs_pad (torch.Tensor): Encoder output [B, T, D]
            hlens (torch.Tensor): Encoder output lengths [B]
            ys_pad (torch.Tensor): Padded target sequences [B, U]
            ali (torch.Tensor): Alignment information
            
        Returns:
            torch.Tensor: SACTC loss value
        """

        # As required by k2, reorder by hlens in descending order
        indices = torch.argsort(hlens, descending=True)
        hlens, hs_pad, ys_pad = hlens[indices], hs_pad[indices], ys_pad[indices]
        if isinstance(ali, torch.Tensor) and ali.dim() == 2:
            ali = ali[indices]
        
        # Compute log probabilities
        nnet_output = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate)) # [B,T,U]
        nnet_output = F.log_softmax(nnet_output, dim=-1)
        
        # Compute per-utterance loss
        loss = self._forward(nnet_output, hlens, ys_pad, ali)

        # Restore the original order
        indices = torch.argsort(indices)
        loss = loss[indices]

        return loss.mean()


    def _forward(
        self, 
        nnet_output: torch.Tensor, 
        hlens: torch.Tensor, 
        ys_pad: torch.Tensor, 
        ali: torch.Tensor
    ) -> torch.Tensor:
        """Internal forward implementation.
        
        Args:
            nnet_output (torch.Tensor): Log probabilities from network
            hlens (torch.Tensor): Input lengths 
            ys_pad (torch.Tensor): Padded target sequences
            ali (torch.Tensor): Alignment information
            
        Returns:
            torch.Tensor: Loss values per utterance
        """
        B, T, D = nnet_output.size()

        # Build supervision tensor for K2
        supervision = torch.stack([
            torch.arange(B),
            torch.zeros(B),
            hlens.cpu()
        ], dim=1).int()

        # Create K2 FSA
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)

        # Process target sequences y
        ys = [[x for x in y if x != -1] for y in ys_pad.cpu().tolist()]
        target_y_idxs = None
        if self.target_y is not None:
            # Mark the position of target special token
            target_y_idxs = torch.Tensor([
                y.index(self.target_y) if self.target_y in y else -1 
                for y in ys
            ]).long().to(nnet_output.device)

        # Build CTC graphs and get output lengths
        ctc_graphs = k2.ctc_graph(ys).to(nnet_output.device)
        olens = torch.Tensor([len(y) for y in ys]).to(nnet_output.device).long()
        U = max(olens)

        # Intersect dense FSA with CTC graph
        lats = k2.intersect_dense(
            ctc_graphs, 
            dense_fsa_vec, 
            BEAM_SIZE,
            seqframe_idx_name="seqframe_idx",
            frame_idx_name="frame_idx",
        )

        # Get arc mappings using _k2
        ragged_lat, arc_map_a, arc_map_b = _k2.intersect_dense(
            a_fsas=ctc_graphs.arcs,
            b_fsas=dense_fsa_vec.dense_fsa_vec,
            a_to_b_map=None,
            output_beam=BEAM_SIZE,
        )

        # Find state indices and scores
        with torch.no_grad():
            state_idx, u_idx, t_idx, fsa_idx = self.find_backward_index(
                ragged_lat, 
                ctc_graphs, 
                dense_fsa_vec, 
                arc_map_a, 
                arc_map_b, 
            )

        # Compute forward and backward scores
        forward_scores  = lats.get_forward_scores(True, True)
        backward_scores = lats.get_backward_scores(True, True)

        # forward
        alpha_ = forward_scores[state_idx]
        alpha = torch.ones([B, U, T]).double().to(nnet_output.device) * float('-inf')
        alpha[fsa_idx, u_idx, t_idx] = alpha_

        # backward
        beta_ = backward_scores[state_idx]
        beta = torch.ones([B, U, T]).double().to(nnet_output.device) * float('-inf')
        beta[fsa_idx, u_idx, t_idx] = beta_

        # fix backward scores
        p_idx = ys_pad[fsa_idx, u_idx].long()
        p_ = nnet_output[fsa_idx, t_idx, p_idx].double()
        p = torch.ones([B, U, T]).double().to(nnet_output.device) * float('-inf')
        p[fsa_idx, u_idx, t_idx] = p_

        beta_prime = log_substraction_exp(
            beta[:, :, :-1], 
            beta[:, :, 1:] + p[:, :, 1:]
        )
        beta_prime = torch.cat([beta_prime, beta[:, :, -1:]], dim=-1)

        # State-level loss
        loss_state = alpha + beta_prime

        # Add risk scores
        if self.risk_factor > 0.0:
            loss_state = loss_state + self.get_risk_scores(
                loss_state, hlens, olens, target_y_idxs
            )

        # Handle invalid loss values
        loss_state = torch.where(
            torch.isnan(loss_state), 
            float('-inf'), 
            loss_state
        )

        # Aggregate scores along time axis
        assert self.aggregate == "time"
        loss_u = torch.logsumexp(loss_state, dim=2)

        if self.den_scale > 0.0:
            loss_u = loss_u - self.den_scale * lats.get_tot_scores(True, True).unsqueeze(1)

        # Fix the invalid loss values
        mask = torch.isinf(loss_u)
        loss_fsas = torch.where(
            mask, 
            0.0, 
            loss_u
        ).sum(1) / (~mask).double().sum(1)

        # Fix the invalid length cases
        loss_fsas = torch.where(hlens < olens, 0.0, loss_fsas)
        if torch.any(hlens < olens):
            print(f"Invalid data: input shorter than output at indices: {(hlens < olens).nonzero()}")
 
        return -loss_fsas 

    def get_risk_scores(
        self, 
        loss_state: torch.Tensor, 
        hlens: torch.Tensor, 
        olens: torch.Tensor, 
        target_y_idxs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute risk scores for alignment control. In SACTC, this is used to 
        control speaker alignment in the output sequence.
        
        Args:
            loss_state (torch.Tensor): Current loss state
            hlens (torch.Tensor): Input lengths
            olens (torch.Tensor): Output lengths  
            target_y_idxs (torch.Tensor, optional): Target token indices
            
        Returns:
            torch.Tensor: Risk scores
        """

        B, U, T = loss_state.size()

        # # Return zero risk if no target tokens
        if not any(idx != -1 for idx in target_y_idxs):
            return torch.zeros_like(loss_state)

        # Calculate proportions, in [B]
        proportion = (target_y_idxs + 1) / olens 
        
        # Calculate time range
        time_range = (torch.arange(1, T + 1, device=loss_state.device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(B, 1, 1))
        hlens_expanded = hlens.unsqueeze(1).unsqueeze(1)
        proportion_expanded = proportion.unsqueeze(1).unsqueeze(1)

        # Calculate risk scores for speaker A and B
        risk_a = -torch.log(self.sigmoid(
            (-time_range / hlens_expanded + proportion_expanded) * -self.risk_factor
        ))

        risk_b = -torch.log(self.sigmoid(
            (time_range / hlens_expanded - proportion_expanded) * -self.risk_factor
        ))

        # Build final risk tensor
        risk = torch.zeros([B, U, T], device=loss_state.device)
        for batch_idx, y_idx in enumerate(target_y_idxs):
            if y_idx != -1:
                num_u_a = y_idx
                num_u_b = U - y_idx - 1

                risk[batch_idx, :y_idx, :] = risk_a[batch_idx].repeat(num_u_a, 1)
                risk[batch_idx, y_idx+1:, :] = risk_b[batch_idx].repeat(num_u_b, 1)
            
        return -risk        

    def find_backward_index(
        self,
        ragged_lat: k2.RaggedTensor,
        a_fsas: k2.Fsa,
        b_fsas: k2.DenseFsaVec,
        arc_map_a: torch.Tensor,
        arc_map_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find states for backward score computation.
        
        Args:
            ragged_lat (k2.RaggedTensor): Ragged lattice
            a_fsas (k2.Fsa): Input FSA
            b_fsas (k2.DenseFsaVec): Dense FSA
            arc_map_a (torch.Tensor): Arc map for FSA a
            arc_map_b (torch.Tensor): Arc map for FSA b
            
        Returns:
            Tuple containing:
                - State indices
                - Token indices 
                - Time indices
                - FSA indices
        """

        # Get the shape of graphs and lattice
        ragint_lat = k2.Fsa(ragged_lat).arcs.shape()
        ragint_graph = a_fsas.arcs.shape()

        # Find arc IDs for incoming arcs of odd-id states in CTC graph
        graph_arc_id_start = ragint_graph.row_splits(2)[1::2]
        graph_arc_id_end = ragint_graph.row_splits(2)[2::2]
        graph_arc_ids = [
            torch.arange(graph_arc_id_start[i], graph_arc_id_end[i])
            for i in range(len(graph_arc_id_end))
        ]
        graph_arc_ids = torch.cat(graph_arc_ids).to(a_fsas.device)

        # Find corresponding arcs in lattice
        start, parts, interval = 0, [], int(2e7 / len(graph_arc_ids))
        while start < len(arc_map_a):
            part_in = arc_map_a[start: min(start+interval, len(arc_map_a))]
            part_out = (graph_arc_ids.unsqueeze(0) == part_in.unsqueeze(1)).int().sum(dim=-1)
            parts.append(part_out)
            start += interval

        lat_arc_ids = torch.cat(parts, dim=0)
        lat_arc_ids = lat_arc_ids.bool().nonzero(as_tuple=True)[0]

        # Find corresponding states in lattice
        state_ids = torch.unique(ragint_lat.row_ids(2)[lat_arc_ids]).long()

        # Get FSA (batch) indices
        fsa_idx = ragged_lat.shape().row_ids(1)[state_ids].long()

        # Calculate time indices
        scores_dim1 = b_fsas.dense_fsa_vec.scores_dim1()
        t_idx = arc_map_b[ragint_lat.row_splits(2)[state_ids].long()] // scores_dim1
        t_idx = (t_idx - b_fsas.dense_fsa_vec.shape().row_splits(1)[:-1][fsa_idx]).long() - 1

        # Calculate token indices
        lat_arc_ids = ragint_lat.row_splits(2)[state_ids].long()
        graph_arc_ids = arc_map_a[lat_arc_ids].long()
        u_idx = (a_fsas.arcs.values()[graph_arc_ids][:, 0] - 1).long() // 2

        return state_ids, u_idx, t_idx, fsa_idx


    def get_constraint_ctc_mask(
        self, 
        ctc_graph: k2.Fsa, 
        dense_fsa_vec: k2.DenseFsaVec, 
        arc_map_a: torch.Tensor, 
        arc_map_b: torch.Tensor, 
        ali: torch.Tensor
    ) -> torch.Tensor:
        """Generate mask for constrained CTC decoding.
        
        Args:
            ctc_graph (k2.Fsa): CTC topology graph
            dense_fsa_vec (k2.DenseFsaVec): Dense FSA from network output
            arc_map_a (torch.Tensor): Arc map for first FSA
            arc_map_b (torch.Tensor): Arc map for second FSA
            ali (torch.Tensor): Alignment information
            
        Returns:
            torch.Tensor: Boolean mask for valid paths
        """
        
        blank_state_u_id = -1 
        safe_threshold = 10000
        num_fsas = len(ctc_graph.arcs.row_splits(1)) - 1

        # Extract destination states for CTC graph arcs
        arc_to_dst = (ctc_graph.as_dict()["arcs"][2 * num_fsas + 4:]
                    .view(-1, 4)[:, 1]
                    .long())
       
        # Calculate token index, for blank states use BLANK_STATE_ID
        u_idx = torch.where(
            arc_to_dst % 2 == 0, 
            blank_state_u_id, 
            (arc_to_dst - 1) // 2
        )

        # Handle ending states
        ctc_ragged = ctc_graph._get_incoming_arcs()
        inverse_shape = ctc_ragged.shape

        # Calculate boundaries for incoming arcs
        incoming_arc_start = inverse_shape.row_splits(2)[
            inverse_shape.row_splits(1)[1:].long() - 1 
        ]
        incoming_arc_end = inverse_shape.row_splits(2)[
            inverse_shape.row_splits(1)[1:].long()
        ]
        # Calculate incoming arcs for each FSA
        incoming_arcs = [
            torch.arange(incoming_arc_start[i], incoming_arc_end[i]) 
            for i in range(num_fsas)
        ]
        incoming_arcs = torch.cat(incoming_arcs).long()
        incoming_arcs = ctc_ragged.values[incoming_arcs].long()
        u_idx[incoming_arcs] = blank_state_u_id

        # Map indices to lattice arcs (u_id: non-blank token id of each lattice arc)
        u_idx = u_idx[arc_map_a.long()]

        # Calculate batch indices (b_idx) of each lattice arc
        fsa_boundaries = ctc_graph.arcs.shape().row_splits(2).long()[
            ctc_graph.arcs.shape().row_splits(1).long()
        ]
        arc_ids = torch.arange(ctc_graph.num_arcs, device=ctc_graph.device)
        b_idx = torch.bucketize(arc_ids, fsa_boundaries, right=True) - 1
        b_idx = b_idx[arc_map_a.long()]

        # Calculate time indices (t_idx) of each lattice arc
        b_shape = dense_fsa_vec.dense_fsa_vec.shape()
        feat_dim = dense_fsa_vec.dense_fsa_vec.scores_dim1()
        duration = dense_fsa_vec.dense_fsa_vec.duration + 1

        t_idx = arc_map_b // feat_dim
        t_shift = torch.zeros(
            1 + len(duration), 
            device=ctc_graph.device, 
            dtype=duration.dtype
        )
        t_shift[1:] = torch.cumsum(duration, dim=0)
        t_idx = t_idx - t_shift[b_idx]

        # Generate constraint mask
        # the non-blank arcs whose t_idx is larger than the time threshold should be kill
        t_threshold = ali[b_idx, u_idx]
        t_threshold = torch.where(
            u_idx == -1, 
            safe_threshold, 
            t_threshold
        )

        return t_idx > t_threshold


def log_substraction_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute log(exp(a) - exp(b)) in a numerically stable way.
    
    Args:
        a (torch.Tensor): First input tensor
        b (torch.Tensor): Second input tensor
        
    Returns:
        torch.Tensor: Result of log(exp(a) - exp(b))
    """
    ans = torch.ones_like(a) * float("-inf")

    # Avoid -inf in input
    mask1 = torch.logical_and(~torch.isinf(a), ~torch.isinf(b))
    a_ = torch.where(mask1, a, -1.0)
    b_ = torch.where(mask1, b, -2.0)

    # Avoid -inf in output: need to be very small as these values would be picked by mask1
    ans_tmp = b_ + ((a_-b_).exp() - 1).log()
    a_ = torch.where(torch.isinf(ans_tmp), -2000.0, a_)
    b_ = torch.where(torch.isinf(ans_tmp), -2001.0, b_)

    ans1 = b_ + ((a_-b_).exp() - 1).log()
    ans = torch.where(mask1, ans1, ans)
 
    # Handle case where only b is infinite
    mask2 = torch.logical_and(~torch.isinf(a), torch.isinf(b))
    ans = torch.where(mask2, a, ans)

    return ans


