import { VersionInfo } from '@start9labs/start-sdk'

/**
 * Initial release. ExVer form `<upstream>:<downstream>` — we track our own
 * packaging revision since LLaMA-Factory has no stable upstream semver.
 */
export const v_0_1_0 = VersionInfo.of({
  version: '0.1.0:0',
  releaseNotes:
    'Initial release: orchestrate LLaMA-Factory LoRA fine-tunes on remote ' +
    'DGX Sparks over SSH, with a web UI for Gemma 4 31B.',
  migrations: {},
})
