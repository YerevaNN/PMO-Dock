from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperienceReplayParameters:
    memory_size: int = 100
    sample_size: int = 10
    smiles: List[str] = field(default_factory=list)
    lead_smiles_in_buffer: Optional[str] = None
    fix_lead_smiles_in_buffer: bool = False