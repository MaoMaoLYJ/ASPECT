# Security

Generated programs and downloaded model code are untrusted inputs. Use disposable
compute containers with restricted networking, no personal credentials, and no
unrelated writable mounts. The Code reward worker's resource limits and process
cleanup are reliability measures, not an adversarial security boundary.

The reference vLLM server enables dynamic adapter loading for local training.
Do not expose this API to an untrusted network. Use scheduler/container network
isolation and bind/access controls appropriate to your cluster. Ray and vLLM
training ports must not be publicly reachable.

Do not commit model assets, datasets, access tokens, private keys, `.env` files,
raw runtime logs, or dataset registries. Run manifests can contain local paths;
review them before sharing. Repository ignore rules are not a substitute for a
privacy review.

Report vulnerabilities through the repository's private security reporting
facility when available. Do not include credentials or sensitive data in public
issues. No platform account or credential is needed by the portable launchers.
