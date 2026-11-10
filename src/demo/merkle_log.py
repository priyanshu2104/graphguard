"""
Append-only log of flag reports with a Merkle-tree summary (RFC 6962 style: SHA-256, 0x00 prefix for
leaves, 0x01 for interior nodes, so a leaf can never be passed off as an interior node).

What it gives you: a single root hash that commits to every record and its order, and a short
inclusion proof showing that one record is part of that root. If any stored record is edited, removed
or reordered, the recomputed root no longer matches a root that was saved earlier.

What it does NOT give you: protection against someone who can rewrite both the records and the saved
roots. Tamper-evidence only holds relative to a root kept somewhere the attacker cannot edit (printed,
emailed, committed elsewhere, or anchored on a public ledger; the ledger anchoring is not implemented).
"""
import hashlib
import json
import time
from pathlib import Path


def _h(data):
    return hashlib.sha256(data).digest()


def leaf_hash(data):
    return _h(b"\x00" + data)


def node_hash(left, right):
    return _h(b"\x01" + left + right)


def _split(n):
    """Largest power of two strictly less than n (n >= 2)."""
    return 1 << ((n - 1).bit_length() - 1)


def merkle_root(leaves):
    n = len(leaves)
    if n == 0:
        return _h(b"")
    if n == 1:
        return leaves[0]
    k = _split(n)
    return node_hash(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def inclusion_proof(leaves, m):
    """Audit path for leaf m (RFC 6962 section 2.1.1)."""
    n = len(leaves)
    if n == 1:
        return []
    k = _split(n)
    if m < k:
        return inclusion_proof(leaves[:k], m) + [merkle_root(leaves[k:])]
    return inclusion_proof(leaves[k:], m - k) + [merkle_root(leaves[:k])]


def verify_inclusion(leaf, m, n, proof, root):
    """Verifies an inclusion proof (RFC 9162 section 2.1.3.2). All arguments are bytes / ints."""
    if m >= n:
        return False
    fn, sn = m, n - 1
    r = leaf
    for p in proof:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def verify_record_hex(record, index, size, proof_hex, root_hex):
    """What an outside party runs: they hold one record, its proof and a published root."""
    return verify_inclusion(leaf_hash(canonical(record)), index, size,
                            [bytes.fromhex(p) for p in proof_hex], bytes.fromhex(root_hex))


class MerkleLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.path.with_name(self.path.stem + "_checkpoints.jsonl")
        self.records = self._read(self.path)

    @staticmethod
    def _read(path):
        if not Path(path).exists():
            return []
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def leaves(self, records=None):
        return [leaf_hash(canonical(r)) for r in (self.records if records is None else records)]

    def append(self, record):
        record = dict(record)
        record["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        return len(self.records) - 1

    def root_hex(self, records=None):
        return merkle_root(self.leaves(records)).hex()

    def proof_hex(self, index):
        return [p.hex() for p in inclusion_proof(self.leaves(), index)]

    def checkpoint(self):
        """Saves the current root. Anchor this value somewhere the log's owner cannot edit."""
        entry = {"size": len(self.records), "root": self.root_hex(),
                 "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def checkpoints(self):
        return self._read(self.checkpoint_path)

    def audit(self, records=None):
        """Recomputes the root of the first `size` records for every saved checkpoint."""
        recs = self.records if records is None else records
        results = []
        for c in self.checkpoints():
            if c["size"] > len(recs):
                results.append({**c, "ok": False, "why": "records were removed"})
                continue
            now = self.root_hex(recs[:c["size"]])
            results.append({**c, "ok": now == c["root"], "why": "" if now == c["root"] else "root differs"})
        return results


def self_test():
    """Exhaustive check of the tree code: every leaf of every tree size up to 70, plus tamper cases."""
    for n in range(1, 71):
        data = [f"record {i}".encode() for i in range(n)]
        leaves = [leaf_hash(d) for d in data]
        root = merkle_root(leaves)
        for m in range(n):
            proof = inclusion_proof(leaves, m)
            assert verify_inclusion(leaves[m], m, n, proof, root), (n, m)
            assert not verify_inclusion(leaf_hash(b"forged"), m, n, proof, root), (n, m, "forged leaf")
            if proof:
                bad = proof[:]
                bad[0] = bytes(32)
                assert not verify_inclusion(leaves[m], m, n, bad, root), (n, m, "bad proof")
            if n > 1:
                assert not verify_inclusion(leaves[m], (m + 1) % n, n, proof, root) or n == 1, (n, m, "wrong index")
    print("merkle self-test passed (trees of 1-70 leaves, every leaf)")


if __name__ == "__main__":
    self_test()