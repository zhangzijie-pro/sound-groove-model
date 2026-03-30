from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from speaker_verification.interfaces.diar_interface import ChunkInferenceResult
from speaker_verification.quantization.turboquant import QuantizedTensorState


@dataclass
class Track:
    global_id: int
    prototype: Optional[torch.Tensor] = None
    prototype_q: Optional[QuantizedTensorState] = None
    hits: int = 1
    misses: int = 0


@dataclass
class TrackerResult:
    local_to_global: Dict[int, int]
    global_frame_ids: torch.Tensor
    active_global_ids: List[int]
    dominant_global_id: Optional[int]


class GlobalSpeakerTracker:
    def __init__(
        self,
        match_threshold: float = 0.72,
        momentum: float = 0.9,
        max_misses: int = 30,
        device: str = "cpu",
        quantizer = None
    ):
        self.match_threshold = float(match_threshold)
        self.momentum = float(momentum)
        self.max_misses = int(max_misses)
        self.device = torch.device(device)
        self.quantizer = quantizer
        
        self.tracks: Dict[int, Track] = {}
        self.next_global_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_global_id = 1

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), p=2, dim=-1)

    @staticmethod
    def _sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        return a @ b.t()

    def _encode_proto(self, proto: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[QuantizedTensorState]]:
        proto = self._norm(proto.detach().to(self.device))
        if self.quantizer is not None:
            return proto, None
        qstate = self.quantizer.quantize(proto)
        return None, qstate
    
    def _decode_proto(self, track: Track) -> torch.Tensor:
        if track.prototype_q is not None:
            proto = self.quantizer.dequantize(track.prototype_q)
            return self._norm(proto.detach().to(self.device))
        if track.prototype is None:
            raise ValueError("Track has neither prototype nor prototype_q")
        return self._norm(track.prototype.detach().to(self.device))
        
    def _new_track(self, proto:torch.Tensor) -> int:
        gid = self.next_global_id
        self.next_global_id += 1
        proto_fp, proto_q = self._encode_proto(proto)
        self.tracks[gid] = Track(
            global_id=gid,
            prototype=proto_fp,
            prototype_q=proto_q,
            hits=1,
            misses=0
        )
        return gid
    
    def _update_track(self, gid: int, proto:torch.Tensor):
        cur = self._norm(proto.detach().to(self.device))
        old = self._decode_proto(self.tracks[gid])
        
        new = self._norm(self.momentum * old + (1.0 - self.momentum) * cur)
        proto_fp, proto_q = self._encode_proto(new)
        
        self.tracks[gid].prototype = proto_fp
        self.tracks[gid].prototype_q = proto_q
        self.tracks[gid].hits += 1
        self.tracks[gid].misses = 0
        
    
    def _age_unmatched(self, matched_gids: set[int]):
        dead = []
        for gid, track in self.tracks.items():
            if gid not in matched_gids:
                track.misses += 1
                if track.misses > self.max_misses:
                    dead.append(gid)
        for gid in dead:
            del self.tracks[gid]
        
        
    @staticmethod
    def _build_global_frame_ids(
        local_frame_ids: torch.Tensor,
        local_to_global: Dict[int, int],
        invalid_value: int = 0,
    ) -> torch.Tensor:
        out = torch.full_like(local_frame_ids.long(), fill_value=invalid_value)
        for local_id, global_id in local_to_global.items():
            out[local_frame_ids == int(local_id)] = int(global_id)
        return out

    def update(self, result: ChunkInferenceResult) -> TrackerResult:
        if len(result.slots) == 0:
            self._age_unmatched(set())
            empty = torch.zeros_like(result.local_frame_ids, dtype=torch.long)
            return TrackerResult({}, empty, [], None)

        local_ids = [int(s.slot) for s in result.slots]
        cur_protos = torch.stack(
            [self._norm(s.prototype.detach().to(self.device)) for s in result.slots], dim=0
        )

        local_to_global: Dict[int, int] = {}
        matched_gids: set[int] = set()

        if len(self.tracks) == 0:
            for slot, proto in zip(result.slots, cur_protos):
                gid = self._new_track(proto)
                local_to_global[int(slot.slot)] = gid
                matched_gids.add(gid)
        else:
            gids = list(self.tracks.keys())
            bank = torch.stack([self.tracks[gid].prototype for gid in gids], dim=0)
            sim = self._sim(cur_protos, bank)
            row_ind, col_ind = linear_sum_assignment((1.0 - sim).detach().cpu().numpy())

            matched_local_idx = set()

            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                score = float(sim[r, c].item())
                if score < self.match_threshold:
                    continue
                gid = gids[c]
                self._update_track(gid, cur_protos[r])
                local_to_global[local_ids[r]] = gid
                matched_gids.add(gid)
                matched_local_idx.add(r)

            for i, slot in enumerate(result.slots):
                if i in matched_local_idx:
                    continue
                gid = self._new_track(cur_protos[i])
                local_to_global[int(slot.slot)] = gid
                matched_gids.add(gid)

        self._age_unmatched(matched_gids)

        global_frame_ids = self._build_global_frame_ids(result.local_frame_ids, local_to_global)
        active_global_ids = sorted(set(global_frame_ids.tolist()) - {0})

        dominant_global_id = None
        if result.dominant_speaker_slot is not None:
            dominant_global_id = local_to_global.get(int(result.dominant_speaker_slot), None)

        return TrackerResult(
            local_to_global=local_to_global,
            global_frame_ids=global_frame_ids,
            active_global_ids=active_global_ids,
            dominant_global_id=dominant_global_id,
        )