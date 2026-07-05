// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title CharterAnchor — governed on-chain anchor for the substrate's 6-root charter.
///
/// The charter (life_precious, self_preservation, promotion_of_intelligence,
/// evolution, correctness, simplicity) is the alignment vocabulary every daemon
/// equilibrates against. It lives off-chain as a Python constant
/// (nodes/common/world_model_substrate/adapter.py, build_charter_world). This
/// contract does NOT store the charter — it anchors WHICH charter version is
/// canonical, by committing the sha256 of the canonical charter blob plus a
/// monotonic version counter. Daemons compare their locally-computed charter
/// hash against currentCharter() and warn loudly on divergence.
///
/// DELIBERATELY GOVERNED — the deliberate contrast with Substrate.sol.
/// -------------------------------------------------------------------
/// Substrate.sol is deliberately UNGOVERNED: it has no admin keys, no owner, no
/// privileged setter — it is purely the attribution layer, and that ungoverned
/// purity is a feature (nobody can rewrite training history). CharterAnchor is
/// its intentional opposite: the charter IS the jurisdiction's core values, and
/// the whole point of a jurisdiction is that GOVERNANCE can change those values.
/// So this contract has exactly one privileged actor — the governor — and one
/// privileged act: anchoring a new charter version. The governor is meant to be
/// a jurisdiction timelock; for now whoever deploys names it.
///
/// FORWARD-ONLY. Charter history is append-only: version n+1 chains to version n
/// via prevHash, mirroring the epoch hash-chain on Substrate.sol. There is no
/// edit, no rollback, no re-basing — a charter change is a forward-only fork
/// boundary (re-basing history would break bit-identical replay, because every
/// historical claim is embedded in the charter's coordinate basis). Old versions
/// stay anchored; the counter only increases.
contract CharterAnchor {
    struct Charter {
        bytes32 hash;   // sha256 of the canonical charter blob
        string uri;     // pointer to the canonical serialization (ipfs/http/sha256-addressed blob)
        bytes32 prevHash; // hash of the previous version (bytes32(0) for version 1)
        uint256 timestamp;
    }

    /// The governor: the only address allowed to anchor a new charter version.
    /// Immutable — governance handoff happens by naming a timelock at deploy.
    address public immutable governor;

    /// version => Charter. Version numbering starts at 1; version 0 is unset.
    mapping(uint256 => Charter) private _charters;

    /// Number of charter versions anchored so far. currentCharter == version count.
    uint256 public versionCount;

    event CharterAnchored(
        uint256 indexed version,
        bytes32 indexed hash,
        string uri,
        bytes32 prevHash
    );

    error NotGovernor();
    error EmptyHash();
    error NoCharter();
    error BadVersion();

    constructor(address governor_) {
        require(governor_ != address(0), "governor=0");
        governor = governor_;
    }

    modifier onlyGovernor() {
        if (msg.sender != governor) revert NotGovernor();
        _;
    }

    /// Anchor a new charter version. Governor-only, append-only: the new version
    /// is versionCount+1 and chains to the current version's hash via prevHash.
    /// @param charterHash sha256 of the canonical charter blob (must be non-zero).
    /// @param uri pointer to the canonical serialization.
    function anchorCharter(bytes32 charterHash, string calldata uri)
        external
        onlyGovernor
    {
        if (charterHash == bytes32(0)) revert EmptyHash();
        uint256 prevVersion = versionCount;
        bytes32 prevHash = prevVersion == 0
            ? bytes32(0)
            : _charters[prevVersion].hash;
        uint256 newVersion = prevVersion + 1;
        _charters[newVersion] = Charter({
            hash: charterHash,
            uri: uri,
            prevHash: prevHash,
            timestamp: block.timestamp
        });
        versionCount = newVersion;
        emit CharterAnchored(newVersion, charterHash, uri, prevHash);
    }

    /// The current (latest) canonical charter. Reverts if none anchored yet.
    function currentCharter()
        external
        view
        returns (uint256 version, bytes32 hash, string memory uri, bytes32 prevHash, uint256 timestamp)
    {
        if (versionCount == 0) revert NoCharter();
        Charter storage c = _charters[versionCount];
        return (versionCount, c.hash, c.uri, c.prevHash, c.timestamp);
    }

    /// A specific charter version. Reverts if version is 0 or beyond versionCount.
    function charterAt(uint256 version)
        external
        view
        returns (bytes32 hash, string memory uri, bytes32 prevHash, uint256 timestamp)
    {
        if (version == 0 || version > versionCount) revert BadVersion();
        Charter storage c = _charters[version];
        return (c.hash, c.uri, c.prevHash, c.timestamp);
    }
}
