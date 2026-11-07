# Verification

- Thirteen local tests passed, including loopback protocol fixtures, recovery, paired comparison and owner-only POSIX permissions for new and updated records with a permissive umask.
- Hermes plugin validation passed without warnings. No model was loaded for these checks.
- Native Windows Python 3.14 passed a local HTTP fixture, verified that a failed response was not automatically retried, and refused recovery while another process held the run lock. No model was loaded and no external service was contacted.
- Unchanged older records are not retroactively chmodded. Actual model evaluation, semantic scoring, production benchmarking, Windows ACL behavior, a reviewed demo and a full publication audit remain unverified.
- 2026-11-07: the local runner was built for the wrong server API (Ollama-only), but the user's models run on llama.cpp servers. It is now \`bakeoff_run_local\`, speaking only the OpenAI-compatible \`/v1/chat/completions\` API (llama.cpp, LM Studio, vLLM; default http://localhost:8080) with \`timeout_seconds\` (10 to 3600) and \`max_tokens\`; all Ollama code was removed. Only the final answer is judged (reasoning_content is ignored); a generation cut off by the token limit is recorded as unknown and not retried. 14 tests pass against an OpenAI-shaped loopback fixture. No real local model has been run through it yet.
