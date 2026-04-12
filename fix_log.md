# Faux_Clawdbot Fix Log

## [2026-04-12] BLK-FCIB-TOONFIX-001 — toon_format PyPI beta constraint

**Package:** `toon_format`  
**Broken constraint:** `toon_format>=0.9.0`  
**Correct constraint:** `toon_format>=0.9.0b1`  

**Root cause:** PEP 440 defines pre-release version semantics. By default, `pip` and version resolvers exclude pre-release versions (alpha, beta, release candidate) unless the constraint explicitly references a pre-release. `0.9.0b1` is a beta release. The constraint `>=0.9.0` does NOT match `0.9.0b1` because pip treats `0.9.0` as a post-release floor that excludes betas. Since no stable `0.9.0` exists on PyPI, the build fails with no matching distribution.

**Fix:** Use `>=0.9.0b1` to explicitly include the beta. This tells pip the floor IS the beta, and it resolves correctly.

**Pattern to apply going forward:** When a package's latest release is a pre-release (alpha/beta/rc), always check PyPI for the actual version string and use that exact pre-release tag in the constraint floor.
