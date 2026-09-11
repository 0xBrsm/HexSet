"""Observable exchange availability, not an estimate of hidden willingness.

One Bernoulli observation per eligible public turn. Batch size and number of
completed deals do not multiply evidence. A shrinking prior and forgetting
limit sparse-history noise and allow changing conditions to be learned.
"""
from dataclasses import dataclass


@dataclass
class TradeActivity:
    prior_success: float = 1.0
    prior_failure: float = 3.0
    half_life: float = 8.0
    successes: float = 0.0
    failures: float = 0.0
    eligible_turns: int = 0
    last_turn: int = -1

    @property
    def mean(self):
        return (self.prior_success+self.successes)/(self.prior_success+self.prior_failure+self.successes+self.failures)

    def observe(self, *, turn, actor, hand_sizes, trade_participants):
        """Public inputs only; no hands, offered gains or gate access.

        Eligibility is deliberately coarse, not proof that a beneficial offer
        existed. It conditions out turns with fewer than two cards for the
        actor or any possible partner. Non-exchange is evidence of low realized
        access, not a claim that somebody refused a beneficial trade.
        """
        if turn <= self.last_turn:
            return
        self.last_turn=turn
        if hand_sizes[actor] < 2 or not any(n>=2 for s,n in enumerate(hand_sizes) if s!=actor):
            return
        decay=2**(-1/self.half_life)
        self.successes*=decay;self.failures*=decay
        succeeded=any(actor in pair for pair in trade_participants)
        self.successes+=int(succeeded);self.failures+=int(not succeeded)
        self.eligible_turns+=1
