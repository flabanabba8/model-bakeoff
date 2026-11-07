# Verification

- Thirteen local tests passed, including loopback protocol fixtures, recovery, paired comparison and owner-only POSIX permissions for new and updated records with a permissive umask.
- Hermes plugin validation passed without warnings. No model was loaded for these checks.
- Native Windows Python 3.14 passed a local HTTP fixture, verified that a failed response was not automatically retried, and refused recovery while another process held the run lock. No model was loaded and no external service was contacted.
- Unchanged older records are not retroactively chmodded. Actual model evaluation, semantic scoring, production benchmarking, Windows ACL behavior, a reviewed demo and a full publication audit remain unverified.
- 2026-11-07: the local runner was built for Ollama only, which the user does not use (their models run on llama.cpp servers). Renamed `bakeoff_run_ollama` to `bakeoff_run_local`; the default API is now OpenAI-compatible (`/v1/chat/completions`, port 8080: llama.cpp, LM Studio, vLLM), with Ollama as `api: ollama`. New `timeout_seconds` (10 to 3600) and `max_tokens`. Only the final answer is judged (reasoning_content is ignored); a generation cut off by the token limit is recorded as unknown and not retried. 16 tests pass, including an OpenAI-shaped loopback fixture.
