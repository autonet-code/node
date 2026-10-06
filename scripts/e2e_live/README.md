# Live E2E harness

Runs the real atn_web app against a real, isolated autonet daemon on a local
hardhat chain. Nothing is mocked on the daemon side; the only stand-in is the
model, which is a deterministic OpenAI-compatible stub.

## Commands

```bash
python scripts/e2e_live/harness.py up       # chain + stub + daemon + fixtures (~15s)
python scripts/e2e_live/harness.py status   # exit 0 only if every piece answers
python scripts/e2e_live/harness.py down     # kill what `up` started, delete the state dir
python scripts/e2e_live/harness.py daemon-stop   # kill ONLY the daemon (A1 kill/restart test)
python scripts/e2e_live/harness.py daemon-start  # relaunch it on the same home and config

python scripts/e2e_live/run_flutter_live.py          # up, flutter test -d windows, down
python scripts/e2e_live/run_flutter_live.py --web    # also chrome via flutter drive (needs chromedriver)
python scripts/e2e_live/run_flutter_live.py --keep-up
```

With the harness up you can run the app tests directly:

```bash
cd C:\code\atn_web
flutter test integration_test/live -d windows --dart-define=DAEMON_WS=ws://127.0.0.1:7799
```

## What `up` starts

| Piece | Where | Notes |
|---|---|---|
| hardhat node | `127.0.0.1:18545`, chain 1337 | started with `node node_modules/hardhat/.../cli.js node`. No `hardhat run`, so `deployments/*.json` is never written |
| contracts | deployed with web3.py from `artifacts/`, signed by hardhat account 0 | Substrate, ServiceRegistry, PaymentChannel, CharterAnchor, VentureVaultFactory |
| stub | `127.0.0.1:18080` (`stub_server.py`) | `POST /v1/chat/completions` (JSON and SSE) replies `E2E-STUB-REPLY <last user message>`; trigger words in the newest user message switch it: `E2E-TOOL` (or `E2E-TOOL:<name>`) makes one tool call then finishes in text, `E2E-STALL` never answers, `E2E-FAIL` returns a 503. `GET /registry.json` is the `ATN_REGISTRY_URL` target and lists local addresses only. `GET /health` |
| daemon | `ws://127.0.0.1:7799` | `python -m atn --headless`. Home is `<state>/home`. Remote and integration listeners are off |

**Fixtures.** All of these are seeded through the daemon's own WS frames:

| Fixture | Details |
|---|---|
| Agents | `e2e-alpha` "E2E Alpha", `e2e-beta` "E2E Beta", and `e2e-alpha-child` "E2E Alpha Child" (parent `e2e-alpha`). All three run on `e2e_stub`/`echo-1`. The first boot also seeds `kevin` |
| Tool | `e2e_echo_upper`: a pinned tool authored by the owner |
| Adoptable tool | `e2e_foreign_lower`: a pinned manifest by the spare account, present only as blobs in the tool store (not registered), so `adopt_tool` + approve works offline. The runner passes its digest as `E2E_ADOPTABLE_DIGEST` |
| On-chain agent | `e2e-alpha` is registered on Substrate, owner-bound to the owner account |
| Service | `E2E Upper Service`: tool-backed, ask 1000. The daemon lists it on the local ServiceRegistry, signed with `e2e-alpha`'s own key |
| Turn check | One `e2e-beta` turn proves the stub answers through the real execution engine |

`up` prints the ws URL, the contract addresses and the funded accounts. All accounts are well-known hardhat keys with 10000 ETH each, valid on the local chain only:

| Role | Address | Use |
|---|---|---|
| deployer | `0xf39F...2266` | deploys contracts, funds gas |
| owner | `0x7099...79C8` | the daemon's `owner_wallet` and `private_key` |
| spare | `0x3C44...93BC` | unused, free for tests (for example a buyer) |

Everything is also written to `<state>/state.json`, together with the logs (`daemon.log`, `hardhat.log`, `stub.log`).

## Configuration

| Env | Default |
|---|---|
| `E2E_LIVE_DIR` | `%TEMP%\atn_e2e_live` (the state dir: home, logs, state.json) |
| `E2E_LIVE_WS_PORT` | `7799` (7700/7701/7710 are refused) |
| `E2E_LIVE_RPC_PORT` | `18545` |
| `E2E_LIVE_STUB_PORT` | `18080` |
| `ATN_WEB_DIR` | `C:\code\atn_web` (runner only) |

## Isolation (why it cannot touch your real daemon or a public chain)

**Home and process isolation**
- The daemon has no `--config` flag, so the harness redirects its home instead. It sets `USERPROFILE` and `HOME` to `<state>/home`, unsets `HOMEDRIVE` and `HOMEPATH`, and pins `KEYSTORE_DIR`.
- The daemon's cwd is the isolated home, not the repo. A `./agents` directory in the cwd would win over the data dir.
- `local_ws_port` is always set. If it is unset, the CLI reclaims 7700 by killing whatever holds it.
- `up` refuses to start when any port it needs is busy, or when the state dir is inside the real `~/.atn`.
- `down` kills only the PIDs it recorded, never by port.

**Chain isolation**
- Every chain field (`rpc_url`, `chain_id`, all contract addresses) is set explicitly in the daemon's config.yaml. Explicit fields are never replaced by the registry, so even a failed `ATN_REGISTRY_URL` fetch cannot pull in Shadownet addresses.
- `up` asserts that the RPC reports chain 1337 before it deploys anything.

**Provider isolation**
- `ATN_DISABLE_PROVIDER_AUTODETECT=1` stops the daemon from probing the Claude Max and Codex bridges. Without it, the harness daemon adopted this machine's Claude Max login even with its home redirected.
- `up` fails if a subscription provider is active.
- `ATN_USEFULNESS_EMBEDDER=hashing` avoids downloading an embedding model into the empty home. `ATN_AUTO_UPDATE=0` and `ATN_INTEGRATION_WS=0` are also set.

**App side** (`atn_web/integration_test/live/live_harness.dart`)
- Refuses a `DAEMON_WS` that is missing, not loopback, or on 7700/7701/7710.
- Uses the isolated URL for both the endpoint and the disconnect fallback.
- Uses `FakeWalletService`, so there is no credential store and no WalletConnect relay.
- Uses in-memory SharedPreferences.
- Asserts that Analytics was never attached (the test pumps `AtnApp` directly, so `main()` and its Firebase block never run).

## Writing live tests

Put tests in `atn_web/integration_test/live/*_test.dart`:
1. Call `IntegrationTestWidgetsFlutterBinding.ensureInitialized()`.
2. Start the app with `pumpLiveApp(tester)`.
3. Wait with `pumpUntilLive` or `pumpUntilFound`. These wait in wall-clock time; `pumpAndSettle` never settles because the background animates forever.
4. Finish with `app.dispose()`.

The fixture ids above are stable. Add new fixtures to `seed()` in `harness.py`.

## Gotchas found while building it

- On Windows, a daemon with stdin redirected to `NUL` exits right after boot. `sys.stdin.isatty()` is True for the NUL device, so `atn/cli.py:773` picks the interactive input loop, reads EOF, and shuts down. The harness passes `--headless` to avoid this.
- The runner runs each live file in its own `flutter test` call. With several files in one call, the Windows device starts the first app and then fails every later one ("Unable to start the app on the device").
- The web variant needs a `chromedriver` on PATH that matches the installed Chrome (Chrome for Testing publishes one per version). It runs `flutter drive -d web-server --browser-name=chrome --browser-dimension=1400,900`. `-d chrome` next to a running chromedriver started the app twice against the same daemon. Each file goes through a temporary `_live_web_entry.dart` at the app root, because the web compiler cannot reach `../../test/support` from a target inside `integration_test/live`.
- The daemon's host scan runs git/gpg with the redirected HOME and can leave a `keyboxd` holding `<home>/.gnupg`. `down` kills any process whose command line names the state dir.
- Adopting `e2e_foreign_lower` installs it for good, so A5's approve path runs once per `up`.
