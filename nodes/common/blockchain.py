"""
Blockchain Interface for Autonet Nodes

Handles interactions with the Autonet chain via Web3.
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransactionResult:
    success: bool
    tx_hash: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class BlockchainInterface:
    """Interface for interacting with the Autonet blockchain."""

    def __init__(
        self,
        rpc_url: str = "http://localhost:8545",
        private_key: Optional[str] = None,
        chain_id: Optional[int] = None,
    ):
        self.rpc_url = rpc_url
        self.private_key = private_key
        self._provided_chain_id = chain_id
        self.chain_id = chain_id or 1337  # Will be auto-detected after connect
        self.web3 = None
        self.account = None

        self._connect()

    def _connect(self) -> None:
        """Establish connection to the blockchain."""
        try:
            from web3 import Web3
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))

            # Auto-detect chain_id from the node if not explicitly provided
            if self.web3.is_connected():
                detected_chain_id = self.web3.eth.chain_id
                if self._provided_chain_id and self._provided_chain_id != detected_chain_id:
                    logger.warning(
                        f"Provided chain_id {self._provided_chain_id} differs from "
                        f"node chain_id {detected_chain_id}. Using node's chain_id."
                    )
                self.chain_id = detected_chain_id
                logger.info(f"Chain ID: {self.chain_id}")

            if self.private_key:
                self.account = self.web3.eth.account.from_key(self.private_key)
                logger.info(f"Connected to blockchain as {self.account.address}")
            else:
                logger.info(f"Connected to blockchain (read-only)")

        except ImportError:
            logger.warning("web3 not installed. Running in mock mode.")
            self.web3 = None
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}")
            self.web3 = None

    def is_connected(self) -> bool:
        """Check if connected to the blockchain."""
        if not self.web3:
            return False
        try:
            return self.web3.is_connected()
        except:
            return False

    def get_balance(self, address: Optional[str] = None) -> int:
        """Get the balance of an address in wei."""
        if not self.is_connected():
            return 0

        addr = address or (self.account.address if self.account else None)
        if not addr:
            return 0

        return self.web3.eth.get_balance(addr)

    def call_contract(
        self,
        contract_address: str,
        abi: list,
        function_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """Call a read-only contract function."""
        if not self.is_connected():
            logger.warning("Not connected to blockchain")
            return None

        contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(contract_address),
            abi=abi,
        )

        func = getattr(contract.functions, function_name)
        return func(*args).call(**kwargs)

    def send_transaction(
        self,
        contract_address: str,
        abi: list,
        function_name: str,
        *args,
        gas_limit: int = 500000,
        **kwargs,
    ) -> TransactionResult:
        """Send a transaction to a contract."""
        if not self.is_connected() or not self.account:
            return TransactionResult(success=False, error="Not connected or no account")

        try:
            contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=abi,
            )

            func = getattr(contract.functions, function_name)
            tx = func(*args).build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'gas': gas_limit,
                'gasPrice': self.web3.eth.gas_price,
                'chainId': self.chain_id,
            })

            signed = self.web3.eth.account.sign_transaction(tx, self.private_key)
            raw = getattr(signed, 'raw_transaction', None) or signed.rawTransaction
            tx_hash = self.web3.eth.send_raw_transaction(raw)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            return TransactionResult(
                success=receipt.status == 1,
                tx_hash=tx_hash.hex(),
                data={"receipt": dict(receipt)},
            )

        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return TransactionResult(success=False, error=str(e))

    def get_block_number(self) -> int:
        """Get the current block number."""
        if not self.is_connected():
            return 0
        return self.web3.eth.block_number

    def get_events(
        self,
        contract_address: str,
        abi: list,
        event_name: str,
        from_block: int = 0,
        to_block: str = "latest",
    ) -> list:
        """Get events from a contract."""
        if not self.is_connected():
            return []

        contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(contract_address),
            abi=abi,
        )

        event = getattr(contract.events, event_name)
        return event.get_logs(from_block=from_block, to_block=to_block)
