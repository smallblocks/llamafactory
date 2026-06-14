import { FileHelper } from '@start9labs/start-sdk'

/**
 * SSH private key used to reach the Sparks, stored as a standalone file in the
 * `main` volume. Kept out of config.json so it is never returned in plaintext
 * config reads. The container copies it to ~/.ssh and chmods 600 at startup.
 *
 * Volume path './ssh/id_spark' -> /data/ssh/id_spark inside the container.
 */
export const sshKeyFile = FileHelper.string('./ssh/id_spark')

/**
 * Optional Hugging Face token (for gated/private model pulls). Gemma 4 is
 * Apache-2.0 so this is usually unnecessary, but kept for private models.
 */
export const hfTokenFile = FileHelper.string('./secrets/hf_token')
