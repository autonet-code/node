"""Platform adapters implementing chat.MessagingClient.

The stub adapter is for tests (no real platform); the discord adapter is the
first real one. Adapters are the only place a platform SDK (discord.py et al.)
is imported — the ChatService stays platform-neutral.
"""
