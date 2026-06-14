import { setupManifest } from '@start9labs/start-sdk'

/**
 * LLaMA-Factory orchestrator manifest.
 *
 * This service does NOT train locally. It is a control plane: a small web UI +
 * SSH client that dispatches LLaMA-Factory fine-tuning jobs to one or two
 * NVIDIA DGX Sparks over the network and streams logs/adapters back. Hence the
 * image is a lightweight, arch-agnostic Python container with no GPU
 * requirement on the StartOS host itself.
 *
 * NOTE: s9pk.mk extracts the package identifier from the single-quoted value on
 * the line below, so keep that field on one line and avoid stray quotes above it.
 */
export const manifest = setupManifest({
  id: 'llama-factory',
  title: 'LLaMA-Factory',
  license: 'Apache-2.0',
  packageRepo: 'https://github.com/ten31/llamafactory',
  upstreamRepo: 'https://github.com/hiyouga/LLaMA-Factory',
  marketingUrl: 'https://github.com/hiyouga/LLaMA-Factory',
  donationUrl: null,
  description: {
    short: 'Fine-tune open LLMs on your DGX Sparks from StartOS',
    long:
      'A StartOS control panel for LLaMA-Factory. Configure a LoRA fine-tune ' +
      '(e.g. Gemma 4 31B) in the browser, and the service dispatches the job ' +
      'over SSH to one or two NVIDIA DGX Sparks, streams training logs back, ' +
      'and retrieves the resulting adapter. No GPU is needed on the StartOS host.',
  },
  // Arch-agnostic orchestrator. Docker build paths are relative to the PROJECT
  // ROOT (where the Makefile runs), matching the Start9 convention.
  images: {
    main: {
      source: {
        dockerBuild: {
          dockerfile: './orchestrator.Dockerfile',
          workdir: '.',
        },
      },
      arch: ['x86_64', 'aarch64'],
      // The orchestrator only SSHes out — it never touches a local GPU.
      nvidiaContainer: false,
    },
  },
  volumes: ['main'],
  dependencies: {},
  hardwareRequirements: {
    ram: 1024,
  },
  alerts: {
    install:
      'This service drives fine-tuning on REMOTE machines (your DGX Sparks) ' +
      'over SSH. After install, open "Configure Sparks" to provide the host ' +
      'address(es) and an SSH key. Nothing trains on your StartOS server.',
  },
})
