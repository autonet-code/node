"""Agent wallet management - economic sovereignty for registered agents."""

from typing import Optional

from .models import AgentWallet


class AgentWalletManager:
    """Manages wallets for registered agents."""

    def __init__(self):
        self._wallets: dict[str, AgentWallet] = {}  # agent_id -> wallet

    def create_wallet(self, agent_id: str, address: str, sponsor_funded: bool = False) -> AgentWallet:
        """Create a wallet for an agent."""
        wallet = AgentWallet(
            address=address,
            sponsor_funded=sponsor_funded,
        )
        self._wallets[agent_id] = wallet
        return wallet

    def get_wallet(self, agent_id: str) -> Optional[AgentWallet]:
        """Get an agent's wallet."""
        return self._wallets.get(agent_id)

    def credit(self, agent_id: str, token_address: str, amount: int, source: str = "") -> bool:
        """Credit tokens to an agent's wallet."""
        wallet = self._wallets.get(agent_id)
        if not wallet:
            return False
        current = wallet.balance.get(token_address, 0)
        wallet.balance[token_address] = current + amount
        wallet.total_earnings += amount
        return True

    def debit(self, agent_id: str, token_address: str, amount: int) -> bool:
        """Debit tokens from an agent's wallet. Returns False if insufficient balance."""
        wallet = self._wallets.get(agent_id)
        if not wallet:
            return False
        current = wallet.balance.get(token_address, 0)
        if current < amount:
            return False
        wallet.balance[token_address] = current - amount
        wallet.total_spent += amount
        return True

    def get_balance(self, agent_id: str, token_address: str) -> int:
        """Get balance of a specific token for an agent."""
        wallet = self._wallets.get(agent_id)
        if not wallet:
            return 0
        return wallet.balance.get(token_address, 0)
